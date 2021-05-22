## README

### Modifications from `timm` source
The model code was taken from `@rwightman`'s `pytorch-image-models` repository, which is called `timm` in PIP ([Github source](https://github.com/rwightman/pytorch-image-models)). There are some things that are not being supported by the library such as patch-wise embedding extraction from Transformer-based models. 

The fork of `timm` is given in `./timm-fork-install/` directory where some modifications are made. Calling `model.encode(x)` will return patch-wise token embeddings once the model is initialized with `timm.create_model` method. For instance,

```
>>> x = torch.randn([1, 3, 224, 224])
>>> model = timm.create_model('vit_base_patch16_224', pretrained=True)
>>> model.encode(x).shape
torch.Size([1, 197, 756])
```

### PyTorch Lightning
In order to reduce the engineering overhead in metric computation, logging, and hyperparameter tuning, `pytorch-lightning` library was used. `pytorch-lightning` can take care of distributed training and autumatic mixed precision (fp-16) training as well. The details on experiment code is given in `main.py` and `modules.py`. 

Note that `modules.py` include `nn.Module` and `pytorch_lightning.LightningModule` object for classification and segmentation. 

## Steps for using this code
1. Install necessary libraries by calling 
```
conda env create -f env.yml
```
2. Download the datasets
```
chmod +x ./scripts/*.sh
# Download BreakHis
./scripts/download_breakhis.sh
./scripts/preprocess_breakhis.sh
# Download CrowdSourcing
./scripts/download_crowdsourcing.sh
```
3. Make changes in `config.yml` to your liking. If you want to add new variables, make sure to add it in appropriate argument `dataclass` object in `arguments.py`. You can also make changes to given variable by simply using `argparse`-like arguments, such as `python main.py --batch_size=16`.

4. Run the code by
```
python main.py
```
