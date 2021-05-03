from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field, InitVar

@dataclass
class BaseArgs:
    seed: int = field(default=42)

    device: str = field(default='cuda')
    amp: bool = field(default=False)

    num_epochs: int = field(default=3)
    batch_size: int = field(default=1)
    
    learning_rate: float = field(default=1e-5)
    eps: float = field(default=1e-8)
    beta_1: float = field(default=0.99)
    beta_2: float = field(default=0.999)
    weight_decay: float = field(default=1e-5)
    
    gradient_clipping: float = field(default=False)
    gradient_max_norm: float = field(default=1.0)
    gradient_accumulation_steps: int = field(default=1)

    scheduler: Optional[str] = field(default=None)
    num_warmup_steps: Optional[int] = field(default=400)

    log_interval: int = field(default=15)
    save_model: bool = field(default=False)
    save_only_best: bool = field(default=True)

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
    data_dir: str = field(default='combined')
    magnification: str = field(default='40X')
    image_size: Tuple[int] = field(default=(460, 700))
    patch_size: Tuple[int] = field(default=(16, 16))
    augmentation: bool = field(default=False)
    train_split: float = field(default=0.7)

    pad_mode: str = field(default='constant')

    model_class: str = field(default='vit_base_patch16_224')
    pretrained: bool = field(default=False)
    checkpoint: Optional[str] = field(default=None)
    num_classes: int = field(default=8)

    attn_drop_rate: float = field(default=0.1)
    drop_rate: float = field(default=0.4)

@dataclass
class SegmentationArgs(BaseArgs):
    data_dir: str = field(default='combined')
    magnification: str = field(default='40X')
    image_size: Tuple[int] = field(default=(460, 800))

    model_class: str = field(default='vit_base_patch16_224')
    pretrained: bool = field(default=False)
    checkpoint: Optional[str] = field(default=None)