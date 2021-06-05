import timm
import torch

from typing import Tuple, Optional

from torch import nn
from einops.layers.torch import Rearrange

import models.transformer.parts

class ViTForClassification(nn.Module):
    """
        Vision Transformer with classification head.
    """
    def __init__(self, args: "Namespace"):
        super().__init__()
        self.encoder = timm.create_model(
            args.model_name,
            img_size=args.image_size,
            pretrained=args.pretrained,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attention_drop_rate
        )
        self.head = nn.Linear(self.encoder.embed_dim, args.num_classes)
        if self.encoder.dist_token is not None:
            self.head_dist = nn.Linear(self.encoder.embed_dim, args.num_classes)

    def forward(self, x):
        x, _ = self.encoder(x)
        if self.encoder.dist_token is None:
            return self.head(x[:, 0])
        else:
            c, d = self.head(x[:,0]), self.head_dist(x[:,1])
            if self.training:
                return c, d
            else:
                return (c + d) * 0.5

class TransformerHead(nn.Module):
    """
        Transformer based segmentation head.
    """
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_tokens = nn.Parameter(torch.randn(1, num_classes, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

    def forward(self, x):
        cls_tokens = self.cls_tokens.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        cls_embed = x[:, :self.num_classes]
        pat_embed = x[:, self.num_classes:]
        return cls_embed, pat_embed

class Upsample(nn.Module):
    """
        Upsampling module.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: Tuple[int, int]
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            Rearrange(
                "b (p1 p2) c -> b c p1 p2",
                p1=image_size[0]//patch_size[0],
                p2=image_size[1]//patch_size[1]
            ),
            nn.Upsample(scale_factor=patch_size, mode='bilinear')
        )
    
    def forward(self, x):
        return self.upsample(x)
    
class ViTForSegmentation(nn.Module):
    """
        Vision Transformer with segmentation head.
    """
    def __init__(
        self,
        args: "Namespace"
    ):
        super().__init__()
        self.encoder = timm.create_model(
            args.model_name,
            img_size=args.image_size,
            pretrained=args.pretrained,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attention_drop_rate
        )
        self.head_type = args.head_type

        if self.head_type == 'convolution':
            self.head = nn.Sequential(
                nn.ConvTranspose2d(16, 32, 4, stride=4),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 2, stride=2),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, 
                    out_channels=args.num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                Rearrange(
                    "b p1 p2 c -> b c p1 p2"
                )                
            )
        elif self.head_type == 'linear':
            self.head = nn.Linear(
                self.encoder.embed_dim,
                args.num_classes
            )
            self.upsample = Upsample(
                image_size=args.image_size,
                patch_size=args.patch_size
            )
        else:
            self.head = TransformerHead(
                num_classes=args.num_classes,
                embed_dim=self.encoder.embed_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads
            )
            self.scale = self.encoder.embed_dim ** 0.5
            self.upsample = Upsample(
                image_size=args.image_size,
                patch_size=args.patch_size
            )


    def forward(self, x):
        _, x = self.encoder(x)
        if self.head_type == 'convolution':
            x = self.head(x)
        elif self.head_type == 'linear':
            x = self.head(x)
            x = self.upsample(x)
        else:
            c, p = self.head(x)
            mask = p @ c.transpose(1,2)
            mask = torch.softmax(mask / self.scale, dim=-1)
            x = self.upsample(mask)
        return x
