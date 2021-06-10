import os
import yaml
from models import JIRNet
from icons_dataset import IconsDataset
from torch.utils.data import DataLoader
from optim_utils import OptimizerWrapper
import torch
import cv2
import numpy as np

def read_ckp(train_output_dir):
    try:
        ckp = open(os.path.join(train_output_dir, 'check_point.txt'))
        start_epoch = int(ckp.readlines()[0])

    except IOError:
        start_epoch = 0

    return start_epoch

def run_eval_epoch(dataset,epoch,net,cfg):
    net.eval()
    worst_mse = torch.zeros(1).cuda()
    for node_id in range(net.images_num):
        # get image
        image_gt = torch.tensor(cv2.imread(dataset.images[node_id]).astype(np.float32)/255.0).cuda()
        image_pred = net.reconstruct_single_image(node_id)
        # calculate image mse
        image_mse = (((image_gt - image_pred)**2).sum(dim = -1)).mean()
        worst_mse = torch.max(image_mse,worst_mse)

    print('maximum_image_mse: %.4f' % (worst_mse.item()))

    loss_log = open(os.path.join(cfg['train_output_dir'], 'eval_loss.txt'), 'a')
    loss_log.write('%d,%.4f\n' % (epoch, worst_mse.item()))
    loss_log.close()



def run_epoch(data_loader, epoch, net, cfg, optimizer_use_model, mode):

    if mode == 'train':
        net.train()

        num_total_iterations = cfg['max_epoch'] * len(data_loader)
        current_iteration = epoch * len(data_loader)
        optimizer_wrapper = OptimizerWrapper({'net': net}, current_iteration, num_total_iterations, cfg)
        if not optimizer_use_model is None:
            optimizer_wrapper.load_optimizer(optimizer_use_model)

    elif mode == 'eval':
        net.eval()
    avg_loss = 0

    for step, curr_data in enumerate(data_loader):

        if mode == 'train':
            current_iteration = epoch * len(data_loader) + step
            optimizer_wrapper.update_lr_and_momentum(current_iteration)
        else:
            optimizer_wrapper = None

        loss = \
            net.run_step(curr_data, optimizer_wrapper)

        if step % cfg['disp_interval'] == 0:
            progress_in_epoch = (step + 1) / len(data_loader)
            print(mode + '_epoch %d, step %d (%.2f %%), '
                  'loss: %.3f' % ( epoch, step, progress_in_epoch * 100, loss))

            if mode == 'train':
                curr_lr, curr_momentun = optimizer_wrapper.get_lr_and_momentum()
                print(mode + '_epoch %d, step %d, lr: %.4f, momentim: %.3f' % (epoch, step, curr_lr, curr_momentun))
        avg_loss += loss

    # epoch_done
    num_samples = len(data_loader)
    avg_loss /= num_samples

    print('avg_loss: %.4f' % (avg_loss))

    loss_log = open(os.path.join(cfg['train_output_dir'], mode + '_loss.txt'), 'a')
    loss_log.write('%d,%.4f\n' % (epoch, avg_loss))
    loss_log.close()

    # save trained weights
    if mode == 'train':
        ckp_epoch = epoch + 1

        net.save_ckp(ckp_epoch,cfg['train_output_dir'])
        optimizer_wrapper.save_ckp(ckp_epoch)

        # update check_point file
        ckp = open(os.path.join(cfg['train_output_dir'], 'check_point.txt'), 'w')
        ckp.write(str(ckp_epoch))
        ckp.close()

def train(cfg):

    os.makedirs(os.path.join(cfg['train_output_dir'],'weights'), exist_ok=True)

    start_epoch = read_ckp(cfg['train_output_dir'])

    dataset_train = IconsDataset(cfg['dataset_dir'], split='train')
    dataset_val = IconsDataset(cfg['dataset_dir'], split='val')

    dataloader_train = DataLoader(dataset_train, batch_size=cfg['batch_size'], num_workers=1, pin_memory=False, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=cfg['batch_size'], num_workers=1, pin_memory=False,
                                  shuffle=False)

    # init model
    net = JIRNet(dataset_val.images_size, dataset_val.images_num)

    # load pretrained weights
    if start_epoch>0:
        net.load_ckp(start_epoch, cfg['train_output_dir'])

    net.cuda()

    optimizer_use_model = None
    if start_epoch>0:
        optimizer_use_model = os.path.join(os.path.join(cfg['train_output_dir'],'weight'),
                                           'Optimizer_{}.pth'.format(start_epoch))

    max_epoch = cfg['max_epoch']
    if cfg['fine_tune']:
        max_epoch = cfg['fine_tune_params']['max_epoch']

    for epoch in range(start_epoch, max_epoch):
        # train epoch
        run_epoch(dataloader_train, epoch, net, cfg, optimizer_use_model, mode = 'train')
        torch.cuda.empty_cache()

        with torch.no_grad():
            run_eval_epoch(dataset_val, epoch, net, cfg)
            # run_epoch(dataloader_val, epoch, net, cfg, optimizer_use_model, mode = 'eval')

        torch.cuda.empty_cache()

if __name__ == '__main__':

    yaml_file_name = 'config.yaml'  # 'half_res_10deg_100cm.yaml' #'RCF360_far_objects_2d_gt.yaml' #'RCF360_psudoGT_baseline.yaml'
    try:
        cfg = yaml.load(open('./' + yaml_file_name, 'r'), Loader=yaml.FullLoader)
    except:
        cfg = yaml.load(open('./' + yaml_file_name, 'r'))

    train(cfg)