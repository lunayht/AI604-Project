import os
import numpy as np
import pandas as pd
from glob import glob
# import cv2
from PIL import Image
# import matplotlib.pyplot as plt
from tqdm import tqdm


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torch.optim as optim
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class CancerDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        super(CancerDataset, self).__init__()
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        mask_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, mask_file)
        image = Image.open(image_path).convert('RGB')
        #         image = np.array(image)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        if self.transform:
            image = self.transform(image)

        label = torch.Tensor(mask).long()
        return image, label




class Reshape(nn.Module):
    def __init__(self, mask_size=1024,num_classes=22):
        super(Reshape, self).__init__()
        self.mask = mask_size
        self.num_classes= num_classes

    def forward(self, x):
        return x.view(-1,self.num_classes,self.mask,self.mask)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            num_classes,
            dim,
            depth,
            heads,
            mlp_dim,
            pool = 'cls',
            channels = 3,
            dim_head = 64,
            dropout = 0.,
            emb_dropout = 0.
        ):
        super().__init__()
        assert image_size[0] % patch_size[0] == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size[0] // patch_size[0])*(image_size[1] // patch_size[1])
        # self. num_patches= num_patches
        patch_dim = channels * patch_size[0]*patch_size[1]
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size[0], p2 = patch_size[1]),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        # self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        # self.deconv_head = nn.Sequential(nn.ConvTranspose2d(16, 32, 4, stride=4),
        #                                  nn.ReLU(),
        #                                  nn.ConvTranspose2d(32, 32, 2, stride=2),
        #                                  nn.ReLU(),
        #                                  nn.Conv2d(in_channels=32, out_channels=num_classes,
        #                                            kernel_size=3, stride=1, padding=1)
        #                                  )
        # self.reshape=Reshape(image_size[0])

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        # x += self.pos_embedding[:, :(n + 1)]
        # x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        # print(x.shape)
        # ch= x.shape[1]
        # print(x.shape)
        # z = x.reshape((-1, ch , 64,64))

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #
        # x = self.to_latent(x)
        # logit =self.deconv_head(z)
        return x[:, 1:]

class MaskTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        emb_dim,
        hidden_dim,
        num_layers,
        num_heads
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_tokens = nn.Parameter(torch.randn(1, num_classes, emb_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers
        )

    def forward(self, x):
        b = x.shape[0]
        cls_tokens = self.cls_tokens.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        c = x[:, :self.num_classes]
        z = x[:, self.num_classes:]
        return c, z

class Upsample(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.upsample = nn.Sequential(
             Rearrange("b (p1 p2) c -> b c p1 p2", p1=image_size//patch_size[0], p2=image_size//patch_size[1]),
             nn.Upsample(scale_factor=patch_size, mode='bilinear')
        )
        # self.p1, self.p2 = image_size//patch_size[0], image_size//patch_size[1] 
        # self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear')
    
    def forward(self, x):
        # print(x.shape)
        # x = x.reshape()
        return self.upsample(x)

class MaskSegmenter(nn.Module):
    def __init__(
        self,
        encoder,
        mask_transformer,
        patch_size,
        image_size,
        emb_dim
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_transformer = mask_transformer
        self.upsample = Upsample(image_size, patch_size)
        self.scale = emb_dim ** -0.5

    def forward(self, x):
         x= self.encoder(x)
         c, z = self.mask_transformer(x)
         masks = z @ c.transpose(1,2)
         masks = torch.softmax(masks / self.scale, dim=-1)
         # print(masks.shape)
         return self.upsample(masks)

from typing import Optional

def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: float = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))

    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros(
        (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
    )
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)


if __name__=='__main__':
    from utils_cs import compute_miou_acc
    # batch_size = 64
    epochs = 200
    lr = 3e-5
    gamma = 0.7
    seed = 42
    device = 'cuda'
    batch_size= 4
    encoder = ViT(
        image_size=(512,512),
        patch_size=(128,128),
        num_classes=22,
        dim=2048,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    )

    mask_ = MaskTransformer(
        num_classes=22,
        emb_dim=2048,
        hidden_dim=1024,
        num_layers=6,
        num_heads=8
    )

    model = MaskSegmenter(
        encoder=encoder,
        mask_transformer=mask_,
        patch_size=(128,128),
        image_size=512,
        emb_dim=2048
    )

    model = nn.DataParallel(model)
    model = model.cuda()
    batch_size = 16
    print('--------------loading data-------------')
    data_dir = './data'
    train_img_dir = os.path.join(data_dir, "train/images")
    train_msk_dir = os.path.join(data_dir, "train/masks")

    test_img_dir = os.path.join(data_dir, "test/images")
    test_msk_dir = os.path.join(data_dir, "test/masks")

    train_img_files = os.listdir(train_img_dir)
    train_msk_files = os.listdir(train_msk_dir)

    test_img_files = os.listdir(test_img_dir)
    test_msk_files = os.listdir(test_msk_dir)

    # print(len(train_img_files), len(test_img_files))

    train_dataset = CancerDataset(train_img_dir, train_msk_dir, transform=transforms.Compose([
        transforms.ToTensor()]))
    print(len(train_dataset))

    test_dataset = CancerDataset(test_img_dir, test_msk_dir, transform=transforms.Compose([
         transforms.ToTensor()]))
    print(len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = DiceLoss() # nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    step_losses = []
    epoch_losses = []

    # epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0
        total=0
        model.train()
        for X, Y in tqdm(train_loader):
            X, Y = X.cuda(), Y.cuda()
            optimizer.zero_grad()
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()
            total += Y.shape[0]
            epoch_loss += loss.item()

        epoch_loss= epoch_loss/total
        print(f" epoch: {epoch} train_loss : {epoch_loss:.4f} \n" )

        acc, iou, dice = compute_miou_acc(model, test_loader)
        print(f" epoch: {epoch} test_iou: {iou:.4f} test_dice: {dice:.4f} test_acc: {acc:.4f} \n")
        
        # epoch_losses.append(epoch_loss / len(train_loader))

    torch.save(model.module.state_dict(), 'pytorchvitunet.pth')

    # X, Y = next(iter(test_loader))
    # X, Y = X.to(device), Y.to(device)
    # Y_pred = model(X)
    # print(Y_pred.shape)
    # Y_pred = torch.argmax(Y_pred, dim=1)
    # print(Y_pred.shape)

    # fig, axes = plt.subplots(16, 3, figsize=(3 * 5, 16 * 5))
    #
    # for i in range(16):
    #     landscape = X[i].permute(1, 2, 0).cpu().detach().numpy()
    #     label_class = Y[i].cpu().detach().numpy()
    #     label_class_predicted = Y_pred[i].cpu().detach().numpy()
    #
    #     axes[i, 0].imshow(landscape)
    #     axes[i, 0].set_title("Landscape")
    #     axes[i, 1].imshow(label_class)
    #     axes[i, 1].set_title("Label Class")
    #     axes[i, 2].imshow(label_class_predicted)
    #     axes[i, 2].set_title("Label Class - Predicted")
