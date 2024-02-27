import torch
import numpy as np
import torch.nn.functional as F

def calc_kldiv(gen_dis_file_root=None, times=10):
    real_dist = []
    realdiv = open('val_methods/realdiv.txt', 'r')
    for line in realdiv:
        items = line.split()
        n = int(items[1])
        real_dist.append(n)
    real_tensor = torch.Tensor(real_dist)
    kl_values = []
    for i in range(times):
        print('\rCalculating %d-th KL divergence.' % i, end='', flush=True)
        file_name = gen_dis_file_root + '/trial_' + str(i) + '/distribution.txt'
        gen_dis_file = open(file_name, 'r')
        gen_dist = []
        for i in range(46656):
            gen_dist.append(0)
        for j, line in enumerate(gen_dis_file):
            items = line.split(',')
            if items.__len__() != 3:
                print("Error in gen distribution! Number %d Line %d" % (i, j))
                continue
            a = int(items[0])
            b = int(items[1])
            c = int(items[2])
            mode = a * 36 * 36 + b * 36 + c
            gen_dist[mode] += 1
        gen_tensor = torch.Tensor(gen_dist)

        kl_value = torch.nn.functional.kl_div(F.log_softmax(gen_tensor), F.softmax(real_tensor), reduction='mean')
        print('KL value %f' % float(kl_value))
        kl_values.append(float(kl_value))

    kls = torch.Tensor(kl_values)
    mean = kls.mean()
    std = kls.std()
    se = std / np.sqrt(times)

    return mean, std, se


if __name__ == '__main__':
    mean, std, se = calc_kldiv(gen_dis_file_root='../results/%s/' % 'results_none', times=1)
    print(mean, std, se)
