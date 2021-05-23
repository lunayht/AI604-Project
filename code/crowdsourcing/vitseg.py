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
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
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

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        # self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        self.deconv_head = nn.Sequential(nn.ConvTranspose2d(16, 32, 4, stride=4),

                                         nn.ReLU(),
                                         nn.ConvTranspose2d(32, 32, 2, stride=2),

                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32, out_channels=num_classes,
                                                   kernel_size=3, stride=1, padding=1)
                                         )
        # self.reshape=Reshape(image_size[0])

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)

        # x += self.pos_embedding[:, :(n + 1)]
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        ch= x.shape[1]
        # print(x.shape)
        z = x.reshape((-1, ch , 64,64))

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #
        # x = self.to_latent(x)
        logit =self.deconv_head(z)
        return logit



if __name__=='__main__':
    # batch_size = 64
    epochs = 200
    lr = 3e-5
    gamma = 0.7
    seed = 42
    device = 'cuda'

    model = ViT(
        image_size=(512,512),
        patch_size=(128,128),
        num_classes=22,
        dim=4096,
        depth=6,
        heads=6,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ).cuda()
    model = nn.DataParallel(model)
    model = model.cuda()
    batch_size = 16
    print('--------------loading data-------------')
    data_dir = './Patches'
    train_img_dir = os.path.join(data_dir, "train/images")
    train_msk_dir = os.path.join(data_dir, "train/masks")

    # test_img_dir = os.path.join(data_dir, "test/images")
    # test_msk_dir = os.path.join(data_dir, "test/masks")

    train_img_files = os.listdir(train_img_dir)
    train_msk_files = os.listdir(train_msk_dir)

    # test_img_files = os.listdir(test_img_dir)
    # test_msk_files = os.listdir(test_msk_dir)

    # print(len(train_img_files), len(test_img_files))

    train_dataset = CancerDataset(train_img_dir, train_msk_dir, transform=transforms.Compose([
        transforms.ToTensor()]))
    print(len(train_dataset))

    # test_dataset = CancerDataset(test_img_dir, test_msk_dir, transform=transforms.Compose([
    #     transforms.ToTensor()]))
    # print(len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)

    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    step_losses = []
    epoch_losses = []

    # epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0
        total=0
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






