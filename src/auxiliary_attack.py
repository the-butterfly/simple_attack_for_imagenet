import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import save_image_with_batch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T


def mean(l): 
    return sum(l)/len(l)

def infer(model, X):
    out = model(X)
    if isinstance(out, tuple):
        return out[0]
    else:  return out

def _atk(models, coeff, X, y, epsilon, niters=100, alpha=0.01, use_mean=False): 
    crit = nn.CrossEntropyLoss(reduction='none')
    X_pgd = Variable(X.data, requires_grad=True)
    
    for i in range(niters): 
        losses = []
        opt = torch.optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        for m, r in zip(models, coeff):
            loss = crit(infer(m, X_pgd), y)
            losses.append(r * loss)

        loss = torch.stack(losses,-1).mean(-1).sum()
        loss.backward()

        if use_mean:
            norm = X_pgd.grad.data.abs().mean(dim=[2,3], keepdim=True)
            eta = alpha * (X_pgd.grad.data / norm)
        else:
            eta = alpha*X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data - eta, requires_grad=True)

        # adjust to be within [-epsilon, epsilon]
        eta = torch.clamp(X.data - X_pgd.data, -epsilon, epsilon)
        X_pgd = Variable(X.data - eta, requires_grad=True)
    return X_pgd.data


def gen_samples_ensemble(name, dataset, models, coeff, epsilon=16/255, niters=20, alpha=1/255, _atk=_atk, use_mean=True, bs=32):
    out_dir = '../data_attacks/'+name+'/images'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    assert len(models) == len(coeff)
    for m in models:
        m.eval()
        for p in m.parameters(): 
            p.requires_grad = False
    
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=4)
    for i, (img,y,z) in enumerate(loader):
        X = Variable(img.cuda(), requires_grad=True)        # center crop 224 from 256
        y, z = y.cuda(), z.cuda()
        y -= 1; z -= 1              # with offset

        X_adv = _atk(models, coeff, X, z, epsilon, niters, alpha, use_mean=use_mean, )
        img_names = dataset.csv.iloc[i*bs : (i+1)*bs, 0].tolist()
        save_image_with_batch(X_adv.cpu(), img_names, out_dir, reverse_transform = T.ToPILImage())
    
    return 
