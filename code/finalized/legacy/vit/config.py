import os
import pathlib
import pyhocon

config_path = pathlib.Path(__file__).parent.absolute()
config = pyhocon.ConfigFactory.parse_file(os.path.join(config_path, 'zoo.conf'))

def drop_head_variant(config):
    try:
        config['representation_size'] = None
    except KeyError:
        pass
    return config

PRETRAINED_MODELS = {
    'B_16': {
        'config': config['b16'],
        'num_classes': 21843,
        'image_size': (224, 224),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth"
    },
    'B_32': {
        'config': config['b32'],
        'num_classes': 21843,
        'image_size': (224, 224),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32.pth"
    },
    'L_16': {
        'config': config['l16'],
        'num_classes': 21843,
        'image_size': (224, 224),
        'url': None
    },
    'L_32': {
        'config': config['l32'],
        'num_classes': 21843,
        'image_size': (224, 224),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32.pth"
    },
    'B_16_imagenet1k': {
        'config': drop_head_variant(config['b16']),
        'num_classes': 1000,
        'image_size': (384, 384),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth"
    },
    'B_32_imagenet1k': {
        'config': drop_head_variant(config['b32']),
        'num_classes': 1000,
        'image_size': (384, 384),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32_imagenet1k.pth"
    },
    'L_16_imagenet1k': {
        'config': drop_head_variant(config['l16']),
        'num_classes': 1000,
        'image_size': (384, 384),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_16_imagenet1k.pth"
    },
    'L_32_imagenet1k': {
        'config': drop_head_variant(config['l32']),
        'num_classes': 1000,
        'image_size': (384, 384),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32_imagenet1k.pth"
    },
}
