import argparse
import os
import numpy as np

from torchvision.utils import save_image

from torch.autograd import Variable

import torch.nn as nn
import torch
from val_methods.modes_rec import calc_modes
from val_methods.kldiv2 import calc_kldiv


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--netG", type=str, default='G.pth', help="path to trained G model")
parser.add_argument("--num_gen", type=int, default=25600, help="how many images to generate")
parser.add_argument("--latent_dim", type=int, default=256, help="dimensionality of the latent space")
parser.add_argument("--method", type=str, default='test', help="to test the models trained with what method")
parser.add_argument("--ntrails", type=int, default=10, help="how many times to test the model")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
out_path = 'stext_results_' + opt.method
model_name = 'G_' + opt.method

os.makedirs("../results", exist_ok=True)
os.makedirs("../results/%s" % out_path, exist_ok=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 7
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 64 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, 2, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 64, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


modes_list = []

for iii in range(opt.ntrails):
    # Initialize generator and discriminator
    generator = Generator()

    if cuda:
        generator.cuda()

    os.makedirs('../results/%s/trial_%d' % (out_path, iii), exist_ok=True)
    opt.netG = f'models/{opt.method}/%s_%d.pth' % (model_name, iii)
    generator.load_state_dict(torch.load(opt.netG))
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for i in range(int(opt.num_gen / opt.batch_size)):
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)
        for j in range(len(gen_imgs.data)):
            print('\rGenerating the ' + str(i * opt.batch_size + j + 1) + '-th image.', end='', flush=True)
            os.makedirs("../results/%s/trial_%d/%d_%d/" % (out_path, iii, i, j), exist_ok=True)
            save_image(gen_imgs.data[j][0], "../results/%s/trial_%d/%d_%d/1.png" % (out_path, iii, i, j), normalize=True)
            save_image(gen_imgs.data[j][1], "../results/%s/trial_%d/%d_%d/2.png" % (out_path, iii, i, j), normalize=True)
            save_image(gen_imgs.data[j][2], "../results/%s/trial_%d/%d_%d/3.png" % (out_path, iii, i, j), normalize=True)

    del generator
    print('')

    modes = calc_modes(times=iii, path='../results/%s/trial_%d' % (out_path, iii))
    print('Modes generated in %d-th trial: ' % iii, modes, '\n')
    modes_list.append(modes)

mean, std, se = calc_kldiv(gen_dis_file_root='../results/%s/' % out_path, times=opt.ntrails)
print(modes_list)
modes = np.array(modes_list)
modes_mean = modes.mean()
modes_std = modes.std()
modes_se = std / np.sqrt(10)
print('Modes:')
print(modes_mean, modes_std, modes_se)
print('KL:')
print(mean, std, se)
