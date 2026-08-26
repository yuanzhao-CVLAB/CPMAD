






<h1 align="center"> CPMAD:Complementary Prototype Mapping for Efficient Multimodal Anomaly Detection </h1>

CPMAD is a multimodal anomaly detection framework that learns shared consensus prototypes and modality-specific supplementary prototypes for RGB-3D/Depth anomaly analysis.


<h2 id="file_cabinet"> :file_cabinet: Datasets </h2>

In our experiments, we employed two datasets featuring rgb images and point clouds: [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) and [Eyecandies](https://eyecan-ai.github.io/eyecandies/). You can preprocess them with the scripts contained in `data/processing_eyecandies.py`(or processing_mvtec_SN). Then, specify your dataset root path using the --root argument and extract data_meta.json.zip.




### :hammer_and_wrench: Setup Instructions

**Dependencies**: Ensure that you have installed all the necessary dependencies. The list of dependencies can be found in the `./requirements.txt` file.


### :rocket: Train CPMAD

Use `train_singleclass.py` in training mode. You can select the classes to be trained by modifying `datasets_classes` in `data/mydataset.py`.

> **Important**: Please set your dataset root path explicitly via `--root`.

```bash
python train_singleclass.py --mode train --root /your/dataset/root
```

You can configure the following options:
   - `--print_bar_step`: Evaluates the model every `print_bar_step` epochs.
   - `--img_size`: Specifies the input image size.
   - `--EPOCHS`: Number of training epochs.
   - `--batchsize`: Number of samples per batch.
   - `--common_codebook_size`: Number of consensus prototypes.
   - `--consensus_prototype_num`: Number of supplementary prototypes used in reconstruction.



### :rocket: Inference CPMAD

Use `train_singleclass.py` in evaluation mode. You can select the classes to be tested by modifying `datasets_classes` in `data/mydataset.py`.

```bash
python train_singleclass.py --mode eval --resume /your/checkpoint/path --root /your/dataset/root
```


## :pray: Contacts

For questions, please send an email to zhaoyuan@mail.dlut.edu.cn.
