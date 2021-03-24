# AI604-Project

## General Idea

We aim to leverage the Transformer architecture to address domain adaptation problem due to CNN's inherent inductive bias. The Transformer architecture has been shown to work in various domains of data formats, including NLP, time series, and most recently, images. 

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

### Repositories
[google-research](https://github.com/google-research/vision_transformer)
[lucidrains](https://github.com/lucidrains/vit-pytorch)
[jeonsworld](https://github.com/jeonsworld/ViT-pytorch)

```
@misc{dosovitskiy2020image,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}, 
      author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
      year={2020},
      eprint={2010.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

We plan to leverage large pre-trained Transformer Encoder to apply to various image domains, possibly combined with CNNs. 

## Some papers

### Medical Image Segmentation 

* [U-Net: Convolution Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) [(Repository)](https://github.com/zhixuhao/unet)
* [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306) [(Repository)](https://github.com/Beckschen/TransUNet)
* 
