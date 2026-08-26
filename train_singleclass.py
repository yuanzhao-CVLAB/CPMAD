import datetime
import glob
import os
import shutil
import sys

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

from random import seed
import json
import numpy as np
from tqdm import tqdm
from itertools import chain
import timm
from data.mydataset import datasets_classes
from timm.scheduler.step_lr import StepLRScheduler

import pandas as pd
import time
from tabulate import tabulate
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from data.metric import cal_metric
from accelerate import Accelerator
from accelerate.logging import get_logger
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur
from typing import List
from torch.optim.lr_scheduler import StepLR

from myutils.loss import BinaryFocalLoss
from myutils.loss_log import LossLog
from mymodels.CPL import CPLTrainer
from data.mydataset import RGBSNDataset
import logging
from torch_optimizer import Lamb

from myutils.StableAdamW import StableAdamW
from myutils.StableAdamW import WarmCosineScheduler

import kornia.filters as K

logger = get_logger(__name__)

device = 'cuda'
kernel2d = K.get_gaussian_kernel2d((15, 15), (4., 4.)).to(device)
kernel2d = kernel2d.squeeze()
base_kernel = kernel2d[None, None, :, :]


def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd


def blur(x, base_kernel=base_kernel):
    B, C, H, W = x.shape
    kernel = base_kernel.expand(C, 1, -1, -1).to(x.device)
    return F.conv2d(x, kernel, padding=7, groups=C)


def print_param_sum(parameters, model):
    total_params = 0
    for param in parameters:
        if param.dtype in [torch.float32, torch.float64, torch.float16]:
            total_params += param.numel()
            param.requires_grad = True
        else:
            print("*" * 10, "not supported dtype ", param.dtype, "*" * 10)
    return total_params / 1e6


def denormlize_img(img):
    img = 255 * img.cpu().mul(torch.tensor([0.485, 0.456, 0.406])[:, None, None]).add_(
        torch.tensor([0.229, 0.224, 0.225])[:, None, None])
    img = F.interpolate(img, size=256, mode="bilinear")
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img_pil = img.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    return img_pil


def train_one_epoch(epoch, iters, trainer, optimizer_rec, lr_scheduler, avg_loss_log, training_dataset_loader):
    tbar = tqdm(training_dataset_loader, disable=True, ncols=150)
    log = ""
    for i, sample in enumerate(tbar):
        iters += i
        loss, loss_str = trainer.train_step(sample, output_dir=args["output_dir"], path=sample["path"], epoch=epoch)

        if args["train_mode"] != "train":
            continue
        optimizer_rec.zero_grad()
        accelerator.backward(loss)
        optimizer_rec.step()
        avg_loss_log.update(loss_str)
        log = 'Epoch:%d, %s lr: %.6f ' % (epoch, avg_loss_log, optimizer_rec.param_groups[0]['lr'])
        tbar.set_description(log)
        lr_scheduler.step()

    logger.info(("*" * 10) + log)
    return iters


def filter_params(model, model_name, args, model_parameters):
    filtered_params = {}
    total_params = 0
    for name, param in model.named_parameters():
        if any(k in f"{model_name}.{name}" for k in args["fine_tune_key"]):
            filtered_params[name] = param
    if len(filtered_params) > 0:
        total_params += print_param_sum(filtered_params.values(), model_name)
        model_parameters.append(list(filtered_params.values()))
    return total_params


def load_parameters(names, models):
    model_parameters = []
    assert len(names) == len(models)
    for model_name, model in zip(names, models):
        total_params = 0
        if isinstance(model, List):
            for p in model:
                total_params += filter_params(p, model_name, args, model_parameters)
        else:
            total_params += filter_params(model, model_name, args, model_parameters)
        logger.info(f'{model_name} Total parameters: {total_params:.2f}M')
    return model_parameters


def train():
    training_dataset = RGBSNDataset(args, train=True)

    training_dataset_loader = DataLoader(training_dataset, batch_size=args['batchsize'], shuffle=True,
                                         num_workers=args["num_workers"], pin_memory=True, drop_last=True)
    print("sample num:", len(training_dataset_loader), len(training_dataset))
    trainer = CPLTrainer(args)

    (names, *models) = trainer.get_models()
    avg_loss_log = LossLog()

    model_parameters = load_parameters(names, models)
    logger.info(f'model_parameters: {sum([sum(p.numel() for p in parameters) for parameters in model_parameters]) / 1e6}M')

    optimizer_model = StableAdamW([{'params': chain(*model_parameters)}],
                                  lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer_model, base_value=1e-3, final_value=1e-4,
                                       total_iters=args["EPOCHS"] * len(training_dataset_loader),
                                       warmup_iters=100)

    if args["train_mode"] != "train":
        args['EPOCHS'] = 1
    tqdm_epoch = range(int(args["resume"].split(os.sep)[-1].split("_")[-1]) + 1 if args["resume"] != "none" else 0,
                       args['EPOCHS'])

    (*models, trainer, training_dataset_loader) = accelerator.prepare(*models, trainer, training_dataset_loader)

    if args["resume"] != "none":
        accelerator.load_state(args["resume"], strict=True)
        logger.info("load chekpoints....:" + args["resume"])

    iters = 0
    if args["mode"] == "eval":
        evalute(trainer)

    for epoch in tqdm_epoch:
        avg_loss_log.reset()
        iters = train_one_epoch(epoch, iters, trainer, optimizer_model, lr_scheduler, avg_loss_log,
                                training_dataset_loader)

        if (epoch + 1) % args["save_checkpoint_epoch"] == 0:
            output_dir = f"{args['output_dir']}/checkpoints/epoch_{epoch}"
            logger.info(f"save checkpoint:{output_dir}")
            accelerator.save_state(output_dir=output_dir)
        if (epoch + 1) % args["print_bar_step"] == 0 or (epoch + 1) == args["EPOCHS"]:
            res = evalute(trainer)
            if res[0, 1] > 0.985:
                return


def to_contiguous_cuda(tensors):
    return [t.to(device).contiguous() if not t.is_contiguous() else t.to(device) for t in tensors]


def pose_preprocess(cos_comb, iter=1):
    if iter == -3:
        return cos_comb, cos_comb
    elif iter == -2:
        cos_comb = blur(cos_comb)
        return cos_comb, cos_comb
    elif iter == -1:
        cos_comb = gaussian_filter(cos_comb.cpu().numpy(), sigma=4)
        cos_comb = torch.from_numpy(cos_comb)
        return cos_comb, cos_comb
    H_W = cos_comb.shape[-1]
    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device=device) / (w_l ** 2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device) / (w_u ** 2)
    cos_comb = cos_comb.reshape(1, 1, H_W, H_W).to(device)
    for _ in range(iter):
        cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
    for _ in range(iter):
        cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
    cos_comb = cos_comb.reshape(H_W, H_W)

    img_score = (cos_comb / torch.sqrt(cos_comb[cos_comb != 0].mean()))
    pix_score = (cos_comb / torch.sqrt(cos_comb.mean()))
    return img_score, pix_score


@torch.no_grad()
def evalute_single(accelerator: Accelerator, dataloader, args, trainer, subclass: str, iter=1):
    device = accelerator.device

    total_image_pred = np.array([])
    total_image_gt = np.array([])
    total_pixel_gt = np.array([])
    total_pixel_pred = np.array([])

    for batch in dataloader:
        target = batch['has_anomaly']
        gt_mask = batch['anomaly_mask']

        _, pred_mask = trainer.eval_step(
            batch, output_dir=args["output_dir"], path=batch["path"][0]
        )

        image_score, pred_mask = pose_preprocess(pred_mask, iter=iter)
        image_score = torch.topk(torch.flatten(image_score), args["max_ratio"], dim=0, largest=True)[0]
        image_score = torch.mean(image_score)
        out_mask = pred_mask = pred_mask[None]

        total_image_pred = np.append(total_image_pred, image_score.detach().cpu().numpy())
        total_image_gt = np.append(total_image_gt, target[0].detach().cpu().numpy())

        flatten_pred_mask = out_mask[0].flatten().detach().cpu().numpy()
        flatten_gt_mask = gt_mask[0].flatten().detach().cpu().numpy().astype(int)

        total_pixel_gt = np.append(total_pixel_gt, flatten_gt_mask)
        total_pixel_pred = np.append(total_pixel_pred, flatten_pred_mask)

    device = accelerator.device
    image_scores = accelerator.gather_for_metrics(torch.tensor(total_image_pred).to(device))
    image_labels = accelerator.gather_for_metrics(torch.tensor(total_image_gt).to(device))
    pixel_scores = accelerator.gather_for_metrics(torch.tensor(total_pixel_pred).to(device))
    pixel_labels = accelerator.gather_for_metrics(torch.tensor(total_pixel_gt).to(device))
    res = cal_metric(image_labels.int(), pixel_labels.int(), image_scores, pixel_scores, args["crop_size"])
    return res


def evalute(trainer):
    res = []
    classes = args['all_datasets_classes']
    bar = tqdm(classes, desc="evalutin ", disable=True)
    for sub_class in bar:
        args['all_datasets_classes'] = sub_class
        bar.desc = f"evalutin class:{sub_class}"
        testing_dataset = RGBSNDataset(args, train=False)
        test_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False, num_workers=args["num_workers"])
        test_loader = accelerator.prepare(test_loader)
        res.append(evalute_single(accelerator, test_loader, args, trainer, sub_class))
    args['all_datasets_classes'] = classes
    total_res.append(res)
    logs = [[c] + list(r) for r, c in zip(res, classes)]
    res = np.array(res)

    logger.info("*" * 90)
    logger.info("*" * 90)
    logger.info("Results--------time:" + str(time.strftime('%Y-%m-%d %H:%M:%S')))
    col_names = ["objects", "Pixel Auroc", "Sample Auroc", "pixel ap", "Sample ap", "pixel aupr", "Sample aupr",
                 "F1max_px", "F1max_sp", "AUPRO@30% ", " AUPRO@10% ", "  AUPRO@5% ", "  AUPRO@1% "]

    class_avg = {}
    for c, r in zip(classes, res):
        data = class_avg.get(c.split("_")[-1], [])
        data.append(r)
        class_avg[c.split("_")[-1]] = data
    all_avg = []
    for key, val in class_avg.items():
        all_avg.append(np.array(val).mean(0).tolist())
        logs.append([f'{key}_avg'] + all_avg[-1])
    logs.append(["All_Class_Mean"] + np.array(res).mean(0).tolist())
    logs.append(["All_Datasets_Mean"] + np.array(all_avg).mean(0).tolist())

    pd_data = pd.DataFrame(logs, columns=col_names)
    new_col_names = ["objects", "Sample Auroc", "Sample aupr", "F1max_sp", "Pixel Auroc",
                     "F1max_px", "AUPRO@30% ", " AUPRO@10% ", "  AUPRO@5% ", "  AUPRO@1% ", "pixel ap", "Sample ap",
                     "pixel aupr"]
    pd_data = pd_data.reindex(columns=new_col_names)

    rename_col = ["Objects", "I-Auroc", "I-Aupr", "I-F1max", "P-Auroc",
                  "P-F1max", "AUPRO@30% ", " AUPRO@10% ", "  AUPRO@5% ", "  AUPRO@1% ", "P-AP", "I-AP", "P-Aupr"]
    logger.info("\n" + tabulate(pd_data.values, headers=rename_col, tablefmt="pipe"))

    return res


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', action='store', type=str, default="outputs/CPLTrainer", required=False)
    parser.add_argument('--root', action='store', type=str,
                        default="/path/to/datasets",
                        required=False)
    parser.add_argument('--batchsize', action='store', type=int, default=10, required=False)
    parser.add_argument('--img_size', action='store', type=int, default=224, required=False)
    parser.add_argument('--EPOCHS', action='store', type=int, default=300, required=False)
    parser.add_argument('--crop_size', action='store', type=int, default=224, required=False)
    parser.add_argument('--resume', action='store', type=str, required=False, default="none")
    parser.add_argument('--print_bar_step', action='store', type=int, default=30, required=False)
    parser.add_argument('--shot', action='store', type=int, default=-1, required=False)
    parser.add_argument('--save_checkpoint_epoch', action='store', type=int, default=1000, required=False)
    parser.add_argument('--num_workers', action='store', type=int, default=4, required=False)
    parser.add_argument('--train_mode', action='store', type=str, default="train", required=False)
    parser.add_argument('--amp', action='store', type=str, default="bf16", required=False)
    parser.add_argument('--lr', type=float, default=1e-3, required=False)
    parser.add_argument('--mode', type=str, default="train", required=False)
    parser.add_argument('--dataset_weight', type=str,
                        default={"EyeRGBNorm": 1, "mvtec3d": 10, "mvtec": 1, "visa1cls": 1, "AD": 1,
                                 "BratsAD1K": 1, "RetinaAD": 1, "LiverAD": 10, "HyperAD": 1}, required=False)
    parser.add_argument('--fine_tune_key', type=str, default=[""], required=False)
    parser.add_argument('--all_datasets', action='store', type=list, default=datasets_classes["unidatasets"],
                        required=False)
    parser.add_argument('--backbone', action='store', type=str, required=False, default="vit_base_patch8_224.dino")
    parser.add_argument('--common_codebook_size', action='store', type=int, default=12, required=False)
    parser.add_argument('--consensus_prototype_num', action='store', type=int, default=8, required=False)
    parser.add_argument('--max_ratio', action='store', type=int, default=65, required=False)

    args = parser.parse_args()
    args_dict = vars(args)

    with open(f'./config.json', 'r') as f:
        args = json.load(f)
    config_dict = defaultdict_from_json(args)
    config_dict.update(args_dict)
    config_dict["img_size"] = (config_dict["img_size"], config_dict["img_size"])
    config_dict["crop_size"] = (config_dict["crop_size"], config_dict["crop_size"])
    seed(42)

    os.makedirs(config_dict["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(config_dict["output_dir"], "logs"), exist_ok=True)

    name = f'{config_dict["output_dir"]}/logs/{datetime.datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]}'
    logger = set_logger(name + ".log")

    file_path = f'{CPLTrainer.__module__.replace(".", os.path.sep)}.py'
    shutil.copyfile(file_path, name + ".py")

    return logger, config_dict


def set_logger(log_file):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not root_logger.hasHandlers():
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    return root_logger


if __name__ == '__main__':
    _, args = parse_args()

    loss_focal = BinaryFocalLoss(reduce=False)
    loss_smL1 = nn.SmoothL1Loss(reduction='none')
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args["amp"]
    )
    blur_func = GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    resume_list = glob.glob(f"{args['output_dir']}/checkpoints/*_epoch_*")
    if len(resume_list) > 0 and args["resume"] == "last":
        args["resume"] = sorted(resume_list, key=lambda x: int(x.split("_")[-1]))[-1]

    total_res = []
    result_last = []

    for index, cls in enumerate(args["all_datasets"]):
        args["all_datasets_classes"] = [cls]
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision=args["amp"]
        )
        if index == 0:
            logger.info(json.dumps(args, indent=4))
        train()
        result_last.append(total_res[-1][0])

    logs = [[c] + list(r) for r, c in zip(result_last, datasets_classes["unidatasets"])]
    res = np.array(result_last)

    logger.info("*" * 90)
    logger.info("*" * 90)
    logger.info("Results--------time:" + str(time.strftime('%Y-%m-%d %H:%M:%S')))
    col_names = ["objects", "Pixel Auroc", "Sample Auroc", "pixel ap", "Sample ap", "pixel aupr", "Sample aupr",
                 "F1max_px", "F1max_sp", "AUPRO@30% ", " AUPRO@10% ", "  AUPRO@5% ", "  AUPRO@1% "]
    class_avg = {}
    for c, r in zip(datasets_classes["unidatasets"], res):
        data = class_avg.get(c.split("_")[-1], [])
        data.append(r)
        class_avg[c.split("_")[-1]] = data
    all_avg = []
    for key, val in class_avg.items():
        all_avg.append(np.array(val).mean(0).tolist())
        logs.append([f'{key}_avg'] + all_avg[-1])
    logs.append(["All_Class_Mean"] + np.array(res).mean(0).tolist())
    logs.append(["All_Datasets_Mean"] + np.array(all_avg).mean(0).tolist())

    pd_data = pd.DataFrame(logs, columns=col_names)
    new_col_names = ["objects", "Sample Auroc", "Sample aupr", "F1max_sp", "Pixel Auroc",
                     "F1max_px", "AUPRO@30% ", " AUPRO@10% ", "  AUPRO@5% ", "  AUPRO@1% ", "pixel ap", "Sample ap",
                     "pixel aupr"]
    pd_data = pd_data.reindex(columns=new_col_names)

    logger.info("\n" + tabulate(pd_data.values, headers=new_col_names, tablefmt="pipe"))