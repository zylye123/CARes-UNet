import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from dataset import MyDataset
from optimizer import RangerV2
from utils import dice_coef_2d
from loss import DiceWithBceLoss
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers in dataloader. In windows, set num_workers=0')
parser.add_argument('--test_img_path', type=str, default=r'.\Test-Image',
                    help='images path for testing')
parser.add_argument('--test_msk_path', type=str, default=r'.\Test-Mask',
                    help='images mask path for testing')

parser.add_argument('--pretrained_model', type=str,default='',
                    help='pretrained base model')

parser.add_argument('--img_save_folder',type=str, default='./CARes_Unet',
                    help='Location to save output images')

parser.add_argument('--save_type', type=str, default='solo',
                    help='Type of saving the result--solo for output while compare for mask and output')





opt = parser.parse_args()



test_transform = transforms.Compose([transforms.ToTensor()])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import CARes_Unet
model = CARes_Unet()


if os.path.exists(opt.pretrained_model):
    model.load_state_dict(torch.load(opt.pretrained_model))
else:
    print("Model Not Found")
    exit(-1)


model.eval()
model = model.to(device)
criterion = DiceWithBceLoss().to(device)
test_loss = 0
test_dice = 0
n = 0

if opt.save_type == 'solo':
    for name in os.listdir(opt.test_img_path):
        img = Image.open(os.path.join(opt.test_img_path,name)).convert('L')
        img = img.resize((256, 256),Image.ANTIALIAS)
        img_msk = Image.open(os.path.join(opt.test_msk_path,name)).convert('L')
        img_msk = img_msk.resize((256, 256),Image.ANTIALIAS)
        img = test_transform(img)
        img_msk = test_transform(img_msk)
        img = img.to(device)
        img_msk = img_msk.to(device)
        img = torch.unsqueeze(img,0)
        img_msk = torch.unsqueeze(img_msk,0)

        output = model(img)
        loss = criterion(output, img_msk)
        dice = dice_coef_2d(output, img_msk)

        test_loss += loss.item()
        test_dice += dice.item()
        n += 1

        output = torch.argmax(output, dim=1, keepdim=True).float()
        output = torch.squeeze(output)
        output = output.cpu().detach().numpy().copy()
        output = output * 255
        output = Image.fromarray(np.uint8(output))




        if not os.path.exists(opt.img_save_folder):
            os.mkdir(opt.img_save_folder)
        output.save(os.path.join(opt.img_save_folder,name))


elif opt.save_type == 'compare':
    for name in os.listdir(opt.test_img_path):
        img = Image.open(os.path.join(opt.test_img_path,name)).convert('L')
        img = img.resize((256, 256),Image.ANTIALIAS)
        img_msk = Image.open(os.path.join(opt.test_msk_path,name)).convert('L')
        img_msk = img_msk.resize((256, 256),Image.ANTIALIAS)
        img = test_transform(img)
        img_msk = test_transform(img_msk)
        img = img.to(device)
        img_msk = img_msk.to(device)
        img = torch.unsqueeze(img,0)
        img_msk = torch.unsqueeze(img_msk,0)

        n += 1
        output = model(img)
        loss = criterion(output, img_msk)
        dice = dice_coef_2d(output, img_msk)
        iter_loss = loss.item()
        test_loss += iter_loss
        test_dice += dice.item()
        output = torch.argmax(output, dim=1, keepdim=True).float()
        output_np = output.cpu().detach().numpy().copy()
        out = output_np[0] * 255

        out = (out).astype(np.uint8)
        out = out[0]

        img_msk = torch.round(img_msk)
        img_msk_np = img_msk.cpu().detach().numpy().copy()
        img_msk_np = img_msk_np[0, 0]

        a = Image.fromarray(np.uint8(out))
        b = Image.fromarray(np.uint8(img_msk_np * 255))

        result = Image.new('L', (256 * 2, 256))
        result.paste(a, box=(0, 0))
        result.paste(b, box=(256, 0))


        if not os.path.exists(opt.img_save_folder):
            os.mkdir(opt.img_save_folder)
        result.save(os.path.join(opt.img_save_folder,name))

print(opt.pretrained_model)
print('test_loss:', test_loss / n)
print('test_dice:', test_dice / n)
