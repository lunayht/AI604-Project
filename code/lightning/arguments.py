from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field, InitVar

@dataclass
class BaseArgs:
    seed: int = field(default=42)

    wandb: bool = field(default=False)

    classification: bool = field(default=True)
    segmentation: bool = field(default=False)

    amp: bool = field(default=False)
    gpus: str = field(default='1')

    max_epochs: int = field(default=3)
    batch_size: int = field(default=1)
    drop_rate: int = field(default=0.0)

    scheduler: bool = field(default=False)

    learning_rate: float = field(default=1e-5)
    eps: float = field(default=1e-8)
    beta_1: float = field(default=0.99)
    beta_2: float = field(default=0.999)
    weight_decay: float = field(default=1e-5)
    
    grad_clipping: float = field(default=False)
    grad_max_norm: float = field(default=1.0)
    accumulate_grad_batches: int = field(default=1)

    warmup_proportion: Optional[float] = field(default=0.15) # Linear scheduler warmsup for first 0.15 training steps

    val_check_interval: float = field(default=0.25) # Evaluate 4 times every epoch
    log_interval: int = field(default=5)
    
    config: InitVar[Dict] = field(default=dict())

    def __post_init__(self, config: Dict):
        for k, v in config.items():
            try:
                if v is not None:
                    self.__setattr__(k, v)
            except AttributeError:
                raise Exception('No config parameter {} found ...'.format(k))

@dataclass
class ClassificationArgs(BaseArgs):
    data_dir: str = field(default='./data/breakhis')
    num_classes_classification: int = field(default=2)

    magnification: str = field(default='40X')
    image_size: Tuple[int] = field(default=(460, 700))
    patch_size: Tuple[int] = field(default=(16, 16))
    pad_mode: str = field(default='constant')
    augmentation: bool = field(default=False)
    train_split: float = field(default=0.7)

    resize: bool = field(default=False)

    model_name: str = field(default='vit_base_patch16_224')
    pretrained: bool = field(default=False)
    checkpoint: Optional[str] = field(default=None)
    num_classes: int = field(default=2)

    attn_drop_rate: float = field(default=0.1)
    drop_rate: float = field(default=0.4)

@dataclass
class SegmentationArgs(BaseArgs):
    data_dir: str = field(default='combined')
    num_classes_segmentation: int = field(default=22)
    
    magnification: str = field(default='40X')
    image_size: Tuple[int] = field(default=(460, 800))

    model_class: str = field(default='vit_base_patch16_224')
    pretrained: bool = field(default=False)
    checkpoint: Optional[str] = field(default=None)
