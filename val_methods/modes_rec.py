import numpy

import warnings
warnings.filterwarnings('ignore')

import os
from vit_pytorch import ViT
import torch
from PIL import Image
import torchvision


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


def evaluate(x, model):

    output = model(x)

    return list(numpy.argmax(output, axis=1))


def calc_modes(times=0, path=None):
    if path is None:
        path = '../results/generated/gdf1o1/0/'
    results_txt = open(path + '/distribution.txt', 'w+')
    list_ = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]

    recorder = []
    v = ViT(
        image_size=32,
        patch_size=16,
        num_classes=36,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        channels=1
    )
    v = torch.nn.DataParallel(v, device_ids=[0])
    v.load_state_dict(torch.load('val_methods/vit.pth'))
    v = v.cuda()
    v.eval()
    for i, folder in enumerate(list_):
        if folder == 'for_kl':
            continue
        if i % 10 == 0:
            print('\rTesting %d-th experiment [%d/%d]' % (times, i, 204800), end='', flush=True)
        subpath = os.path.join(path, folder)
        img1_path = os.path.join(subpath, '1.png')
        img2_path = os.path.join(subpath, '2.png')
        img3_path = os.path.join(subpath, '3.png')
        to_tensor = torchvision.transforms.ToTensor()
        norm = torchvision.transforms.Normalize([0.5], [0.5])
        img1 = norm(to_tensor(Image.open(img1_path).convert('L').resize((32,32))).unsqueeze(0)).cuda()
        img2 = norm(to_tensor(Image.open(img2_path).convert('L').resize((32,32))).unsqueeze(0)).cuda()
        img3 = norm(to_tensor(Image.open(img3_path).convert('L').resize((32,32))).unsqueeze(0)).cuda()

        a = v(img1)
        a = torch.argmax(a, dim=1, keepdim=False)
        a1 = 10000 * int(a[0])
        b = v(img2)
        b = torch.argmax(b, dim=1, keepdim=False)
        b1 = 100 * int(b[0])
        c = v(img3)
        c = torch.argmax(c, dim=1, keepdim=False)
        c1 = 1 * int(c[0])
        r = a1 + b1 + c1

        if not (r in recorder):
            recorder.append(r)
        results_txt.write(str(int(a[0]))+','+str(int(b[0]))+','+str(int(c[0]))+'\n')

    del v
    results_txt.close()
    return recorder.__len__()


if __name__ == '__main__':
    root = ''
    modes_list = []
    for i in range(1):
        folder = str(i+1)
        path = os.path.join(root, folder)
        modes_list.append(calc_modes(times=i, path=path))
    print(modes_list)
