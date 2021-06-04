# Vision Transformer for Breast Cancer Classification and Segmentation

This repository contains the code for using Vision Transformer (ViT) for breast cancer classification and semantic segmentation. 

## Instructions
The repository has the following file structure:

```bash
├── env.yml
├── config_cls.yml
├── config_seg.yml
├── README.md
├── arguments.py
├── classify.py
├── segment.py
├── loader.py
├── kill_zombie.sh
├── download
│   ├── download_cls.py
│   ├── download_data.sh
│   ├── download_seg.py
│   ├── preprocess_seg.py
│   └── train_test_split.py
├── dataset
│   ├── breakhis
│   └── crowdsourcing
├── legacy
│   ├── classify_accelerate.py
│   ├── config.yml
│   └── test.py
└── models
    ├── transformer
    │   ├── parts.py
    │   └── vit.py
    └── unet
        ├── parts.py
        └── unet.py



```

### Setting up the environment
The code base is entirely built on PyTorch Lightning, which provides a reproducible, scalable framework for research. The necessary packages are included in `env.yml`. Modify `cudatoolkit` version to appropriate version for your machine. You can setup the environment with `conda` by simply

```bash
conda env create -f env.yml
```
### Downloading the dataset
The repository is has two parts, mainly classification and segmentation. In order to download the datasets at once, run the script at `download/download_data.sh`. This downloads the data to `dataset` directory under respective dataset name.

### Models
Vision Transformer and related models are all taken from `pytorch-image-models` or `timm` library (https://github.com/rwightman/pytorch-image-models) to easily port publicly available pretrained weights. Since the library implementation does not output patch-wise encoded representations, the class was overriden with inherited implementation under `models/transformer` directory.
### Configurations
Training configurations can be modified with `config_cls.yml` and `config_seg.yml` for classification and segmentation respectively. PyTorch Lightning should take care of distributed training (`ddp` (distributed data parallel) is the default distributed backend). Depending on your machine's GPU architecture, automatic mixed precision (AMP) training is supported.

## Legacy codes
Some of the legacy codes can be found under `legacy` directory.

## Notes
Keyboard-interrupting the training will lead to lingering DDP processes that need to be killed manually by running `kill_zombie.sh`.

## Dataset Citations
```
@article{spanhol2015dataset,
  title={A dataset for breast cancer histopathological image classification},
  author={Spanhol, Fabio A and Oliveira, Luiz S and Petitjean, Caroline and Heutte, Laurent},
  journal={Ieee transactions on biomedical engineering},
  volume={63},
  number={7},
  pages={1455--1462},
  year={2015},
  publisher={IEEE}
}

@article{amgad2019structured,
  title={Structured crowdsourcing enables convolutional segmentation of histology images},
  author={Amgad, Mohamed and Elfandy, Habiba and Hussein, Hagar and Atteya, Lamees A and Elsebaie, Mai AT and Abo Elnasr, Lamia S and Sakr, Rokia A and Salem, Hazem SE and Ismail, Ahmed F and Saad, Anas M and others},
  journal={Bioinformatics},
  volume={35},
  number={18},
  pages={3461--3467},
  year={2019},
  publisher={Oxford University Press}
}
```

## Library Citations
```
@article{falcon2019pytorch,
  title={PyTorch Lightning},
  author={Falcon, WA, et al.},
  journal={GitHub. Note: https://github.com/PyTorchLightning/pytorch-lightning},
  volume={3},
  year={2019}
}

@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}

@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

