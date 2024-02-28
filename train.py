import argparse
import os
import numpy as np
import math

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch
from dataset import StackedText, StackedText_test, StackedText_static

os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--latent_dim", type=int, default=256, help="dimensionality of the latent space")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--method", type=str, default='test',
                    help="method to suppress mode collapse: none, test")
parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--use_tqdm", type=bool, default=True, help="use tqdm to show progress bar for calculating density function")
parser.add_argument("--kdt_num_intervals", type=int, default=51, help="how many intervals to split pixel values")
parser.add_argument("--save_real_density", type=str, default=None, help="save calculated real density to pt file")
parser.add_argument("--read_real_density", type=str, default=None, help="read calculated real density from pt file")
parser.add_argument("--ntrails", type=int, default=10, help="train the models for how many times")

opt = parser.parse_args()
print(opt)
if opt.use_tqdm:
    from tqdm import tqdm

os.makedirs("models/" + opt.method, exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        X = 1

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.BatchNorm2d(out_filters), nn.LeakyReLU(0.3, inplace=True)]
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, int(8*X)),
            *discriminator_block(int(8*X), int(16*X)),
            *discriminator_block(int(16*X), int(32*X)),
        )

        self.adv_layer = nn.Sequential(nn.Linear(int(32*X)*4*4, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


adversarial_loss = torch.nn.BCELoss()
df_loss = torch.nn.L1Loss()

real_density = torch.zeros((opt.kdt_num_intervals, opt.channels, opt.img_size, opt.img_size)).cuda()
density_fetched = False

for iii in range(opt.ntrails):
    # Initialize generator and discriminator

    avg_times = 0
    total_mean = 0
    total_var = 0
    total_std = 0

    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        df_loss.cuda()

    datatset = StackedText_static(size=opt.img_size)
    dataloader = torch.utils.data.DataLoader(dataset=datatset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.n_cpu)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    def calc_freq(imgs):
        global real_density
        real_imgs = Variable(imgs.type(Tensor))*255
        zero_tensor = torch.zeros_like(real_imgs)
        one_tensor = torch.ones_like(real_imgs)
        interval_len = math.ceil(255/opt.kdt_num_intervals)
        real_qi = []
        for i in range(opt.kdt_num_intervals):
            current_interval = (interval_len * i, interval_len * i + interval_len)
            where_greater = torch.where(real_imgs >= current_interval[0], one_tensor, zero_tensor)
            where_less = torch.where(real_imgs < current_interval[1], one_tensor, zero_tensor)
            qi = torch.mul(where_greater, where_less)
            real_qi.append(qi.squeeze(0))
        real_qi = torch.stack(real_qi, dim=0)
        real_density += real_qi

    def calc_density():
        global real_density
        real_density = real_density / test_datatset.__len__()

        interval_len = math.ceil(256 / opt.kdt_num_intervals)
        real_density = real_density / interval_len
        real_density[-1, :, :, :] = real_density[-1, :, :, :] * (interval_len - 1) / (
                    255 - interval_len * (opt.kdt_num_intervals - 1))


    if (opt.method == 'test') and (not density_fetched):
        if opt.read_real_density is not None:
            real_density = torch.load(opt.read_real_density)
            density_fetched = True
        else:
            test_datatset = StackedText_test(size=opt.img_size)
            test_loader = torch.utils.data.DataLoader(dataset=test_datatset,
                                                     batch_size=1,
                                                     num_workers=0)
            with torch.no_grad():
                if opt.use_tqdm:
                    for i, imgs in tqdm(enumerate(test_loader), total=test_datatset.__len__()):
                        calc_freq(imgs)
                else:
                    for i, imgs in enumerate(test_loader):
                        print(f'Processing {i}/{test_datatset.__len__()} image to calculate the density function...',
                              end='', flush=True)
                        calc_freq(imgs)
                calc_density()
                density_fetched = True
            if opt.save_real_density is not None:
                torch.save(real_density, opt.save_real_density)

    for epoch in range(opt.n_epochs):
        for i, imgs in enumerate(dataloader):
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            adv_loss = g_loss.clone()

            #############################
            if opt.method == 'test':
                gen_imgs_for_kdt = (gen_imgs + 1) * 255 / 2
                one_tensor = torch.zeros_like(gen_imgs_for_kdt).fill_(0)
                interval_len = math.ceil(255 / opt.kdt_num_intervals)
                fake_qi = []
                max_stat = []
                for ii in range(opt.kdt_num_intervals):
                    current_interval = (interval_len * ii, interval_len * ii + interval_len)
                    u = (current_interval[1] + current_interval[0]) / 2
                    sigma = 3
                    where_less = torch.exp(-0.5 * ((gen_imgs_for_kdt - u) / sigma) ** 10)
                    Fexp = where_less.sum(dim=0) / gen_imgs_for_kdt.shape[0]
                    Freal = torch.zeros((opt.channels, opt.img_size, opt.img_size)).cuda()
                    for jj in range(ii+1):
                        Freal += real_density[jj, :, :, :] * interval_len
                    test_stat = torch.abs(Freal-Fexp)
                    max_stat.append(test_stat)
                max_stat = torch.stack(max_stat, dim=0)
                max_stat = torch.max(max_stat, dim=0)[0]
                test_loss = torch.sum(max_stat)
                if epoch >= 0:
                    g_loss = g_loss + 0.1 * test_loss
            else:
                test_loss = torch.Tensor([-1])
            ##############################

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2.

            d_loss.backward()
            optimizer_D.step()
            if i % 10 == 0:
                print(
                    "[Trial %d/%d] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Test loss: %f]"
                    % (iii, opt.ntrails, epoch + 1, opt.n_epochs, i, len(dataloader), d_loss.item(), adv_loss.item(),
                       test_loss.item()
                       )
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

        if epoch % 50 == 0:
            torch.save(generator.state_dict(), 'models/checkpoints/G_%d.pth' % epoch)
            torch.save(discriminator.state_dict(), 'models/checkpoints/D_%d.pth' % epoch)

    torch.save(generator.state_dict(), 'models/' + opt.method + '/G_' + opt.method + '_' + str(iii) + '.pth')
    torch.save(discriminator.state_dict(), 'models/' + opt.method + '/D_' + opt.method + '_' + str(iii) + '.pth')
