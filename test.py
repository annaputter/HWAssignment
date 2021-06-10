import os
import yaml
from models import JIRNet
from icons_dataset import IconsDataset
from torch.utils.data import DataLoader
from optim_utils import OptimizerWrapper
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_ckp(train_output_dir):
    try:
        ckp = open(os.path.join(train_output_dir, 'check_point.txt'))
        start_epoch = int(ckp.readlines()[0])

    except IOError:
        start_epoch = 0

    return start_epoch

def plot_representation(cfg):
    net = JIRNet((48, 48), 100)
    net.load_ckp(read_ckp(cfg['train_output_dir']), cfg['train_output_dir'])
    net.eval()
    # get final layer weights
    with torch.no_grad():
        weights_matrix = net.get_outermost_layers_weights()

    plt.imshow(weights_matrix[:, :, 0], cmap='jet')
    plt.show()

    plt.imshow(weights_matrix[:, :, 1], cmap='jet')
    plt.show()

    plt.imshow(weights_matrix[:, :, 2], cmap='jet')
    plt.show()

def reconstruct_images(cfg):
    # load net
    net = JIRNet((48, 48), 100)
    net.load_ckp(read_ckp(cfg['train_output_dir']), cfg['train_output_dir'])
    net.cuda()
    net.eval()
    os.makedirs(os.path.join(cfg['train_output_dir'],'experiments','exp_1'),exist_ok=True)
    with torch.no_grad():
        for n in range(100):
            I = net.reconstruct_single_image(n).cpu().numpy()
            cv2.imwrite(os.path.join(cfg['train_output_dir'],'experiments','exp_1','%00d.png'%(n)),(I*255).astype(np.uint8))

def upsample_images(cfg):
    # load net
    net = JIRNet((48, 48), 100)
    net.load_ckp(read_ckp(cfg['train_output_dir']), cfg['train_output_dir'])
    net.cuda()
    net.eval()
    os.makedirs(os.path.join(cfg['train_output_dir'], 'experiments', 'exp_2'), exist_ok=True)
    with torch.no_grad():
        for n in range(100):
            I = net.upsample_single_image(n,256).cpu().numpy()
            cv2.imwrite(os.path.join(cfg['train_output_dir'], 'experiments', 'exp_2', '%00d.png' % (n)),
                        (I * 255).astype(np.uint8))

def interpolate_image_pairs(cfg):
    # load net
    net = JIRNet((48, 48), 100)
    net.load_ckp(read_ckp(cfg['train_output_dir']), cfg['train_output_dir'])
    net.cuda()
    net.eval()

    os.makedirs(os.path.join(cfg['train_output_dir'], 'experiments', 'exp_3'), exist_ok=True)

    with torch.no_grad():
        # find 3 best pairs
        weights_matrix = net.get_outermost_layers_weights()
        dist_matrix = np.zeros((net.images_num,net.images_num))
        for n in range(net.images_num):
            dist_matrix[n,:] = ((weights_matrix[n:n+1,:,:] - weights_matrix)**2).sum(axis=2).sum(axis=1)
            dist_matrix[n,n] = np.inf

        min_val = dist_matrix.min(axis = 1)
        min_idx = dist_matrix.argmin(axis = 1)
        sort_idx = min_val.argsort()

        pairs = [(sort_idx[0],min_idx[sort_idx[0]]),(sort_idx[2],min_idx[sort_idx[2]]), (sort_idx[4],min_idx[sort_idx[4]])]

        # generate images
        for pair in pairs:
            I = net.interpolate_image_pairs(pair[0],pair[1],0.5).cpu().numpy()
            cv2.imwrite(os.path.join(cfg['train_output_dir'], 'experiments', 'exp_3', '%00d_%00d.png' % (pair[0], pair[1])),
                        (I * 255).astype(np.uint8))

if __name__ == '__main__':

    yaml_file_name = 'config.yaml'  # 'half_res_10deg_100cm.yaml' #'RCF360_far_objects_2d_gt.yaml' #'RCF360_psudoGT_baseline.yaml'
    try:
        cfg = yaml.load(open('./' + yaml_file_name, 'r'), Loader=yaml.FullLoader)
    except:
        cfg = yaml.load(open('./' + yaml_file_name, 'r'))


    if cfg['experiment_num']==0:
        plot_representation(cfg)
    elif cfg['experiment_num']==1:
        reconstruct_images(cfg)
    elif cfg['experiment_num']==2:
        upsample_images(cfg)
    elif cfg['experiment_num'] == 3:
        interpolate_image_pairs(cfg)