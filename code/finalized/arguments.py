from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, InitVar

@dataclass
class BaseArgs:
    """Common arguments for both subtasks"""
    gpus: str = field(default='1')
    workers: int = field(default=8)
    seed: int = field(default=42)
    checkpoint: str = field(default='')

    amp: bool = field(default=False)
    amp_backend: str = field(default='native')
    distributed: bool = field(default=True)

    wandb: bool = field(default=False)
    log_interval: int = field(default=5)
    val_check_interval: float = field(default=0.5) # Evaluate 2 times every epoch

    image_size: Tuple[int] = field(default=(460, 700))
    patch_size: Tuple[int] = field(default=(32, 32))
    num_classes: int = field(default=2) # Number of classes
    train_split: float = field(default=0.7)
    
    accumulate_grad_batches: int = field(default=1)
    grad_clipping: float = field(default=False)
    grad_max_norm: float = field(default=1.0)

    max_epochs: int = field(default=3)
    batch_size: int = field(default=1)
    attention_drop_rate: float = field(default=0.0)
    drop_rate: int = field(default=0.0)

    learning_rate: float = field(default=1e-5)
    eps: float = field(default=1e-8)
    beta_1: float = field(default=0.99)
    beta_2: float = field(default=0.999)
    weight_decay: float = field(default=1e-5)
    warmup_proportion: float = field(default=0.15)

    model_name: str = field(default='vit_base_patch32_384')
    pretrained: bool = field(default=False)
    
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
    magnification: str = field(default='40X')

    pad_mode: str = field(default='constant')
    augmentation: bool = field(default=False)
    resize: bool = field(default=False)

@dataclass
class SegmentationArgs(BaseArgs):
    data_dir: str = field(default='./data/crowdsourcing')
    head_type: str = field(default='transformer')
    hidden_dim: int = field(default=512)
    num_layers: int = field(default=6)
    num_heads: int = field(default=8)
