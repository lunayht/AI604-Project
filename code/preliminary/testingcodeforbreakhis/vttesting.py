import os
import numpy as np
import pandas as pd
from glob import glob
# import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import shutil

# In[7]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPUs index

batch_size = 64

# In[9]:


# train_transforms = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ]
# )

# val_transforms = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ]
# )


# test_transforms = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ]
# )


# In[ ]:


# In[10]:


transform = transforms.Compose([transforms.Resize((460, 700)),
                                transforms.RandomResizedCrop((460, 700)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose([transforms.Resize((460, 700)),
                                     #                                 transforms.RandomResizedCrop((460,700)),
                                     #                                 transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root='./data/data_40X/Mtrain/',
                                            transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='./data/data_40X/Mtest/',
                                           transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

# In[11]:


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# # get some random training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# # print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


# In[ ]:


# In[12]:


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size[0] % patch_size[0] == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# In[13]:


# batch_size = 64
epochs = 500
lr = 3e-5
gamma = 0.7
seed = 42

# In[14]:


model = ViT(
    image_size=(460, 700),
    patch_size=(20, 50),
    num_classes=8,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
).cuda()

model = nn.DataParallel(model)
model = model.cuda()

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# In[ ]:


best_acc = 0
model_dir = 'pyvitcheckpoint40x.pth'
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    avg_loss = 0.
    train_acc = 0.
    correct = 0
    total = 0
    model.train()
    for data, label in tqdm(train_loader):
        data = data.cuda()
        label = label.cuda()

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

        _, predicted = torch.max(output.data, 1)

        total += label.size(0)
        train_acc += (predicted == label).sum().item()

    epoch_loss = avg_loss / total
    epoch_acc = 100.0 * train_acc / total
    print(f'Epoch: {epoch + 1}/{epochs} Loss: {epoch_loss:.4f} '
          f' train acc:{epoch_acc:.2f}')

    with torch.no_grad():
        model.eval()
        test_accuracy = 0
        test_loss = 0
        correct = 0
        total = 0
        # test_loss = 0
        for data, label in test_loader:
            data = data.cuda()
            label = label.cuda()

            output = model(data)
            test_loss = criterion(output, label)

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    test_loss += test_loss / total
    test_acc = 100 * correct / total
    print(f"test_loss : {test_loss:.4f} - acc: {100 * correct / total:.4f}\n"
          )
    if best_acc < test_acc:
        best_acc = test_acc
        torch.save(model.module.state_dict(), model_dir)

print("Best accuracy: " + str(best_acc))


