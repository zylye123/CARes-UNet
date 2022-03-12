import os

import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
from datetime import datetime
from dataset import MyDataset
from optimizer import RangerV2
from utils import dice_coef_2d
from loss import DiceWithBceLoss
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=1201,
                    help='epoch number')
parser.add_argument('--start_iter', type=int, default=1, 
                    help='Starting Epoch')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.1,
                    help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30,
                    help='every n epochs decay learning rate')
parser.add_argument('--decay_start_epoch', type=int, default=800,
                    help='when to start to decay learning rate')
parser.add_argument('--batchsize', type=int, default=2,
                    help='training batch size')
parser.add_argument('--data_augmentation', type=bool, default=False)

parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers in dataloader. In windows, set num_workers=0')
parser.add_argument('--train_img_path', type=str, default=r'.\Train-Image',
                    help='images path for training')
parser.add_argument('--train_msk_path', type=str, default=r'.\Train-Mask',
                    help='images mask path for training')
parser.add_argument('--semi_img_path', type=str, default=r'',
                    help='images path for training with fake labels')
parser.add_argument('--semi_msk_path', type=str, default='Pos-Mask',
                    help='images mask for training with fake labels')
parser.add_argument('--optimizer_type', type=str, default='Ranger',
                    help='type of optimizer')
parser.add_argument('--pretrained', type=bool, default=False)

parser.add_argument('--pretrained_model', type=str,default='', 
                    help='pretrained base model')
parser.add_argument('--save_start_epoch', type=int, default=200,
                    help='starting to save model epoch ')
parser.add_argument('--snapshots', type=int, default=30,
                    help='every n epochs save a model')

parser.add_argument('--save_folder',type=str, default='./', 
                    help='Location to save checkpoint models')

opt = parser.parse_args()

if opt.data_augmentation:
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(90),
                                        transforms.ToTensor()
                                        ])
else:
    train_transform = transforms.Compose([transforms.ToTensor()])


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

train_dataset = MyDataset(opt.train_img_path,opt.train_msk_path,train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers)
semi_dataset = MyDataset(opt.semi_img_path, opt.semi_msk_path, train_transform)
semi_dataloader = cycle(DataLoader(semi_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers))


epo_num = opt.epoch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from model import CARes_Unet
model = CARes_Unet()
model = model.to(device)


if opt.pretrained:
    if os.path.exists(opt.pretrained_model):
        model.load_state_dict(torch.load(opt.pretrained_model, map_location=lambda storage, loc: storage))
        print('Pre-trained model is loaded.')
    else:
        print("Model Not Found")
        exit(-1)


criterion = DiceWithBceLoss().to(device)
if opt.optimizer_type == 'Ranger':
    optimizer = RangerV2(model.parameters(),lr = opt.lr)
elif opt.optimizer_type == 'Adam':
    optimizer = optim.Adam(model.parameters(),lr = opt.lr)
elif opt.optimizer_type == 'SGD':
    optimizer = optim.SGD(model.parameters(),lr = opt.lr)
print('optimizer:',opt.optimizer_type)

prev_time = datetime.now()

for epo in range(opt.start_iter, epo_num):
    train_loss = 0
    train_dice = 0
    model.train()
    for index, (img, msk) in enumerate(train_dataloader):
        img = img.to(device)
        msk = msk.to(device)

        semi_img, semi_msk = next(semi_dataloader)
        semi_img = semi_img.to(device)
        semi_msk = semi_msk.to(device)

        img = torch.cat((img, semi_img), 0)
        msk = torch.cat((msk, semi_msk), 0)

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, msk)
        dice = dice_coef_2d(output, msk)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_dice += dice.item()

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    prev_time = cur_time

    print('epoch: %f, train loss = %f, train dice = %f ,%s'
          % (epo, train_loss / len(train_dataloader), train_dice / len(train_dataloader), time_str))

    if epo % opt.decay_epoch == 0 and epo > opt.decay_start_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= opt.decay_rate
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if epo >opt.save_start_epoch and epo %opt.snapshots==0:
        save_path = opt.save_folder + 'semi_supervised_' + opt.model_type + '_epoch_{}.pth'.format(epo)
        torch.save(model.state_dict(), save_path)
        print("model_copy is saved !")








