import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from preprocess import LABELS

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def plot_confusion_matrix(logits: "torch.Tensor", labels: "torch.Tensor", num_classes: int):
    label_dict = {'B': 0, 'M': 1}
    if num_classes == 8:
        label_dict = {k: v for k,v in LABELS.items()}

    preds = torch.argmax(logits, dim=-1).tolist()
    labels = labels.squeeze(-1).tolist()

    matrix = np.zeros((num_classes, num_classes))
    for lbl, pre in zip(labels, preds):
        matrix[lbl, pre] += 1
    matrix /= matrix.sum(axis=1)[:,None]

    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['figure.dpi'] = 80

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("proportion", rotation=-90, va='bottom')

    keys = list(label_dict.keys())
    ax.set_xticks(np.arange(len(label_dict)))
    ax.set_yticks(np.arange(len(label_dict)))
    ax.set_xticklabels(keys)
    ax.set_yticklabels(keys)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('True Labels')

    for i in range(len(keys)):
            for j in range(len(keys)):
                text = ax.text(j, i, round(matrix[i,j], 3), ha='center', va='center', color='w')

    ax.set_title('Confusion Matrix')
    return fig, ax

def compute_metric(logits: "torch.Tensor", labels: "torch.Tensor", num_classes: int):
    """Computes validation metrics according to number of labels"""
    assert len(logits) == len(labels)
    tot = len(logits)
    if num_classes > 5:
        top3 = torch.topk(logits, k=3, dim=-1)[1]
        top1 = top3[:,:1]

        top3 = torch.sum(top3 == labels).cpu().item()
        top1 = torch.sum(top1 == labels).cpu().item()
        return {'acc@3': top3 / tot, 'acc@1': top1 / tot}
    else:
        top1 = torch.topk(logits, k=1, dim=-1)[1]
        top1 = torch.sum(top1 == labels).cpu().item()
        return {'acc@1': top1 / tot}