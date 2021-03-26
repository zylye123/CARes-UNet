import os
from PIL import Image
from torch.utils.data import Dataset
import random
import torch

class MyDataset(Dataset):
    def __init__(self,img_path,mask_path, transform=None):
        self.transform = transform
        self.img_path = img_path
        self.mask_path = mask_path
    def __len__(self):
        return len(os.listdir(self.img_path))

    def __getitem__(self, idx):
        img_name = os.listdir(self.img_path)[idx]
        img = Image.open(os.path.join(self.img_path,img_name)).convert('L')
        img = img.resize((256, 256),Image.ANTIALIAS)
        seed = random.randint(0, 4294967295)
        random.seed(seed)
        img = self.transform(img)    
        img_msk = Image.open(os.path.join(self.mask_path,img_name)).convert('L')
        img_msk = img_msk.resize((256, 256),Image.ANTIALIAS)
        random.seed(seed)
        img_msk = self.transform(img_msk)
        img_msk = torch.round(img_msk)
        return img, img_msk


class SingleDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem(self, idx):
        image = Image.oepn(self.image_path[idx]).convert('L')

        if self.transform:
            image = self.transform(image)

        return image