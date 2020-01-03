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


def mean(l): 
    return sum(l)/len(l)

def infer(model, X):
    out = model(X)
    if isinstance(out, tuple):
        return out[0]
    else:  return out

def _atk(model, X, y, epsilon, niters=100, alpha=0.01, use_mean=False): 
    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters): 
        opt = torch.optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(infer(model, X_pgd), y)
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

def target_attack(loader, model, epsilon, niters=100, alpha=0.01, verbose=False,  _atk=_atk, use_mean=False, offset=True):
    model.eval()
    total_err, total_succ = [],[]
    if verbose: 
        print("Requiring no gradients for parameters.")
    # for p in model.parameters(): 
    #     p.requires_grad = False
    
    for i, (X,y,z) in enumerate(loader):
        X,y = Variable(X.cuda(), requires_grad=True), Variable(y.cuda().long())
        z = Variable(z.cuda().long())
        if offset:
            y -= 1; z -= 1
        X_pgd = _atk(model, X, z, epsilon, niters, alpha,use_mean)
        preds = model(X_pgd).data.argmax(1)
        error = (preds != y.data).sum().item()  / X.size(0)
        success = (preds == z.data).sum().item() / X.size(0)
        total_err.append(error)
        total_succ.append(success)

        if verbose: 
            print('success: {} | attack: {}'.format(error, success))
    
    print('[TOTAL] error: {} | success: {}'.format(mean(total_err), mean(total_succ)))
    
    return mean(total_err), mean(total_succ)

def gen_samples_with_resize(name, dataset, model, epsilon=16/255, niters=20, alpha=1/255, _atk=_atk, use_mean=True, beg=0, offset=True, bs=32, **kwargs):
    out_dir = '../data_attacks/'+name+'/images'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    model.eval()
    for p in model.parameters(): 
        p.requires_grad = False
    
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=4)

    for i, (img,y,z) in enumerate(loader):
        if i < beg:  continue               # for continue, for debug, 
        X = Variable(img.cuda(), requires_grad=True)        # center crop 224 from 256
        y, z = y.cuda(), z.cuda()
        if offset:
            y -= 1; z -= 1

        X_adv = _atk(model, X, z, epsilon, niters, alpha, use_mean=use_mean)
        img_names = dataset.csv.iloc[i*bs : (i+1)*bs, 0].tolist()
        save_image_with_batch(X_adv.cpu(), img_names, out_dir, **kwargs)
    
    return 

