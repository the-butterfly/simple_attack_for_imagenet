import os
import sys
from PIL import Image
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset as Dataset
from torch.utils.data.dataloader import DataLoader as DataLoader
import torchvision.transforms as T
import dill
import torch as ch
# import matplotlib.pyplot as plt

mean = ch.tensor([0.485, 0.456, 0.406])
std = ch.tensor([0.229, 0.224, 0.225])

class myData(Dataset):
    def __init__(self, im_dir='../data', transform=T.ToTensor()):
        # super().__init__()
        self.file = os.path.join(im_dir, 'dev.csv')
        assert os.path.isfile(self.file)
        self.csv = pd.read_csv(self.file)
        self.im_dir = os.path.join(im_dir, 'images')
        assert os.path.isdir(self.im_dir)
        self.nsamples = len(self.csv)
        self.transform = transform

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        fpath = os.path.join(self.im_dir, self.csv.iloc[index][0])
        img = Image.open(fpath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        lab = self.csv.iloc[index][1]
        target = self.csv.iloc[index][2]
        return img, lab, target



def input_diversity(x,image_width,image_resize,prob=0.5):
    rnd = np.random.randint(image_width,image_resize)
    rescaled = torch.nn.functional.interpolate(x,[rnd,rnd], mode='bilinear',align_corners=True)
    
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    
    pad_top = np.random.randint(0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem)
    pad_right = w_rem - pad_left
    
    padded = torch.nn.functional.pad(rescaled,\
            (pad_left,pad_right,pad_top,pad_bottom))
    
    if np.random.random()>prob:
        ret = padded
    else:
        ret = x
    
    image = torch.nn.functional.interpolate(ret,[image_width,image_width], mode='bilinear',align_corners=True)
    
    return image  


class InpModel(nn.Module):
    def __init__(self, model, mean=mean, std=std, tomax=330, p=0.5):
        super().__init__()
        self.smax = tomax
        self.p = p
        self.model = model
        new_std = std[..., None, None]
        new_mean = mean[..., None, None]
        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)
        
    def forward(self, x):
        x = ch.clamp(x, 0, 1)
        x = input_diversity(x, 299, self.smax, self.p)
        x = torch.nn.functional.interpolate(x,[224,224], mode='bilinear',align_corners=True)
        x_normalized = (x - self.new_mean)/self.new_std
        out = self.model(x_normalized)
        return out

    
def resume_madry_model(model, resume_path, parallel=True):
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = ch.load(resume_path, pickle_module=dill)

        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]:v for k,v in sd.items()}
        model.load_state_dict(sd)
        if parallel:
            model = ch.nn.DataParallel(model)
        model = model.cuda()

        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    else:
        print("=> pretrained model doesn't exist!")
    model.eval()
    return model, checkpoint


def save_image_with_batch(tensor, img_name, out_dir, reverse_transform=T.ToPILImage()):
    assert len(tensor) == len(img_name), "Tensor length {}, img_name list length {}".format(tensor, img_name)
    for i in range(tensor.size(0)):
        img = reverse_transform(torch.clamp(tensor[i], 0, 1))
        img.save(os.path.join(out_dir, img_name[i]))
    

def test(model, data_loader, test_crit=nn.CrossEntropyLoss(), offset=True, target=False, training=False):
    with torch.no_grad():
        if training: model.train()
        else: model.eval()
        loss1 = 0; cor1 = 0; cor2 = 0
        for batch_idx, (data, label, attack) in enumerate(data_loader):
            data, label, attack = data.cuda(), label.cuda(), attack.cuda()
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            # add a blank class 1001
            if offset:
                nll_logit = torch.zeros_like(label, dtype=torch.float32)[:,None]
                output = torch.cat([nll_logit, output], dim=1)
            tloss1 = test_crit(output, label).item()
            loss1 += tloss1
            
            preds = output.argmax(-1)
            cor1 += preds.eq(label.view_as(preds)).sum().item()
            cor2 += preds.eq(attack.view_as(preds)).sum().item()
            sys.stdout.write('step: %04d/%04d | loss1: %.4f |\r' % (batch_idx, len(data_loader), tloss1))
            sys.stdout.flush()
        loss1 /= len(data_loader)
        cor1 *= 100. / len(data_loader.dataset)
        cor2 *= 100. / len(data_loader.dataset)
        cls_str = '                                                   \n'
        loss1_str = 'Test set: Avg loss: %.4f | ' % (loss1)
        acc_str = 'Accuracy: %.2f | ' % (cor1)
        if target:
            acc_str += 'attack rate: %.2f | ' % (cor2)
        print(cls_str+loss1_str+' '+acc_str+'\n')
    return loss1, cor1, cor2

