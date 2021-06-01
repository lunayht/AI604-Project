import torch
import timm.models.vision_transformer

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    VisionTransformer

        A modified version of timm's ViT that overrides the
        original implementation to return [CLS], [DIST]
        token embeddings and patch embeddings as tuples.
    """
    def forward_features(self, x):
        """
        Original implementation:
        
            x = self.norm(x)
            if self.dist_token is None:
                return self.pre_logits(x[:, 0])
            else:
                return x[:, 0], x[:, 1]
        """
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x)
        else:
            return x

    def forward(self, x):
        """
        Original implementation:
        
            x = self.forward_features(x)
            if self.head_dist is not None:
                x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
                if self.training and not torch.jit.is_scripting():
                    # during inference, return the average of both classifier predictions
                    return x, x_dist
                else:
                    return (x + x_dist) / 2
            else:
                x = self.head(x)
            return x
        """
        x = self.forward_features(x)
        if self.dist_token is None:
            return x[:,:1], x[:,1:]
        else:
            return x[:,:2], x[:,2:]

timm.models.vision_transformer.VisionTransformer = VisionTransformer