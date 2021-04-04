import os
from glob import glob
from dataset import SingleDataset
from utils import dice_coef_2d
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.vgg import VGG
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image
from datetime import datetime
from model import *
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_type', type=str, default='U_net',
                     help='type of model')
parser.add_argument('--model_path', type=str, default=r'',
                     help='path of supervised model')
parser.add_argument('--Image_dir', type=str, default=r'',
                     help='path of images to be labeled')
parser.add_argument('--Mask-dir', type=str, default='test_img_pre',
                   help='path of masks to be saved')

opt = parser.parse_args()

def cycle(iterable):
    while True:
        print('end')
        for x in iterable:
            yield x

Image_path = glob(os.path.join(opt.Image_dir, '*'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(opt.model_type)

transform = transforms.Compose([
    transforms.ToTensor(),
])

if opt.model_type == 'Res_Unet':
    model = CARes_Unet()
else:
    print('Model Not Found')
    exit(-1)

if os.path.exists(opt.model_path):
    model.load_state_dict(torch.load(opt.model_path))
else:
    print('Model Not Found!')
    exit(-1)



model.eval()
model.to(device)

test_loss = 0
test_dice = 0
i = 1
if not(os.path.exists(opt.Mask_dir)):
    os.mkdir(opt.Mask_dir)


for name in os.listdir(opt.Image_dir):

    img = Image.open(os.path.join(opt.Image_dir,name)).convert('L')
    img = img.resize((256,256),Image.ANTIALIAS)

    img = transform(img)
    img = img.view((1, img.shape[0], img.shape[1], img.shape[2]))
    img = img.to(device)
    output = model(img)
    output = torch.argmax(output, dim=1, keepdim=True).float()
    output_np = output.cpu().detach().numpy().copy()
    out = output_np[0] * 255

    out = (out).astype(np.uint8)
    out = out[0]
    out_img = Image.fromarray(out)
    out_img.save(os.path.join(opt.Mask_dir,name))










