import os
import torch
from PIL import Image
import random
import torchvision


class StackedText(torch.utils.data.Dataset):
    def __init__(self, root='data/ICDAR13_chars', size=28):
        super(StackedText, self).__init__()
        self.image_list = []
        folders = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
        for folder in folders:
            names = [x for x in os.listdir(os.path.join(root, folder)) if x .endswith('jpg')]
            for name in names:
                self.image_list.append(os.path.join(root, folder, name))
        self.size = size

    def __getitem__(self, index):
        rs = [random.randint(0, self.image_list.__len__() -1), random.randint(0, self.image_list.__len__()-1), random.randint(0, self.image_list.__len__()-1)]
        img1 = Image.open(self.image_list[rs[0]]).convert('L').resize((self.size, self.size), resample=Image.BILINEAR)
        img2 = Image.open(self.image_list[rs[1]]).convert('L').resize((self.size, self.size), resample=Image.BILINEAR)
        img3 = Image.open(self.image_list[rs[2]]).convert('L').resize((self.size, self.size), resample=Image.BILINEAR)
        to_tensor = torchvision.transforms.ToTensor()
        img1 = to_tensor(img1)
        img2 = to_tensor(img2)
        img3 = to_tensor(img3)
        img = torch.cat((img1, img2, img3), dim=0)
        norm = torchvision.transforms.Normalize([0.5], [0.5])
        img = norm(img)
        return img

    def __len__(self):
        return len(self.image_list)*13


class StackedText_test(torch.utils.data.Dataset):
    def __init__(self, root='data/ICDAR13_chars', size=28):
        super(StackedText_test, self).__init__()
        self.image_list = []
        folders = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
        for folder in folders:
            names = [x for x in os.listdir(os.path.join(root, folder)) if x .endswith('jpg')]
            for name in names:
                self.image_list.append(os.path.join(root, folder, name))
        self.size = size

    def __getitem__(self, index):
        rs = [random.randint(0, self.image_list.__len__() -1), random.randint(0, self.image_list.__len__()-1), random.randint(0, self.image_list.__len__()-1)]
        img1 = Image.open(self.image_list[rs[0]]).convert('L').resize((self.size, self.size), resample=Image.BILINEAR)
        img2 = Image.open(self.image_list[rs[1]]).convert('L').resize((self.size, self.size), resample=Image.BILINEAR)
        img3 = Image.open(self.image_list[rs[2]]).convert('L').resize((self.size, self.size), resample=Image.BILINEAR)
        to_tensor = torchvision.transforms.ToTensor()
        img1 = to_tensor(img1)
        img2 = to_tensor(img2)
        img3 = to_tensor(img3)
        img = torch.cat((img1, img2, img3), dim=0)
        return img

    def __len__(self):
        return len(self.image_list)*13
