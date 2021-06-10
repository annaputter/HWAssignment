import torch
import numpy as np
import h5py

import os

class LinearDecaySchedule():
    def __init__(self, total_steps, max_lr, div = 10, momentum = 0.9):
        self.start_val = max_lr
        self.final_val = max_lr / div
        self.num_steps = total_steps
        self.momentum = momentum

    def calc(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.), self.momentum

class OneCycleScheduler():
    def __init__(self, total_steps, max_lr=0.01, momentum_vals=(0.95, 0.85), prcnt=0.4, div=10, ann_div=1e2):
        self.total_steps = total_steps  # total number of iterations including all epochs
        self.div = div  # the division factor used to get lower boundary of learning rate
        self.first_step_len = int(self.total_steps * prcnt)
        self.high_lr = max_lr  # the optimum learning rate, found from LR range test
        self.low_lr = max_lr / div
        self.high_mom = momentum_vals[0]
        self.low_mom = momentum_vals[1]
        self.annihilation_div = ann_div

    def calc(self, iteration):  # calculates learning rate and momentum for the batch
        lr = self.calc_lr(iteration)
        mom = self.calc_mom(iteration)
        return lr, mom

    def annealing_cos(self, start, end, pct):
        # print(pct, start, end)
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = np.cos(np.pi * pct) + 1
        return end + (start - end) / 2 * cos_out

    def annealing_linear(self, start, end, pct):
        # print(pct, start, end)
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return start + (end - start) * pct

    def calc_lr(self, iteration):

        if iteration < self.first_step_len:
            lr = self.annealing_cos(self.low_lr, self.high_lr, iteration / self.first_step_len)
        else:
            lr = self.annealing_cos(self.high_lr, self.low_lr / self.annihilation_div,
                                    (iteration - self.first_step_len) /
                                    (self.total_steps - self.first_step_len))
        return lr

    def calc_mom(self, iteration):

        if iteration < self.first_step_len:

            mom = self.annealing_cos(self.high_mom, self.low_mom, iteration / self.first_step_len)
        else:
            mom = self.annealing_cos(self.low_mom, self.high_mom,
                                    (iteration - self.first_step_len) /
                                    (self.total_steps - self.first_step_len))
        return mom


class OptimizerWrapper:
    def __init__(self, net_dict, iteration, num_total_iterations, cfg):

        self.fine_tune = cfg['fine_tune']
        self.fine_tune_params = cfg['fine_tune_params']
        self.train_output_dir = cfg['train_output_dir']
        self.opt_param = cfg['opt_param']
        self.scheduler = LinearDecaySchedule(total_steps=num_total_iterations, max_lr=cfg['max_lr'])
        self.optimizer = self.get_optimizer(net_dict, iteration, cfg)

    def get_optimizer(self, nets, iteration, cfg):

        if self.fine_tune:
            lr, momentum = self.fine_tune_params['lr'], self.fine_tune_params['momentum']
        else:
            lr, momentum = self.scheduler.calc(iteration)

        weight_decay = cfg['weight_decay']

        params = []
        if cfg['opt_param']=='all':
            for net_name in nets:
                params.append({'params': nets[net_name].parameters()})
        else:
            params.append({'params': nets[cfg['opt_param']].parameters()})

        optimizer = None
        if cfg['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(params=params,
                                        momentum=momentum,
                                        weight_decay=weight_decay,
                                        nesterov=True,
                                        lr=lr)

        elif cfg['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(params=params,
                                         betas=(momentum, 0.999),
                                         weight_decay=weight_decay,
                                         lr=lr)

        elif cfg['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(params=params,
                                          betas=(momentum, 0.999),
                                          weight_decay=weight_decay,
                                          lr=lr)

        assert optimizer is not None

        return optimizer

    def get_lr_and_momentum(self):
        return self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['betas'][0]

    def update_lr_and_momentum(self, iteration):

        if self.fine_tune:
            lr, momentum = self.fine_tune_params['lr'], self.fine_tune_params['momentum']
        else:
            lr, momentum = self.scheduler.calc(iteration)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer.param_groups:
            param_group['betas'] = (momentum, 0.999)
        return

    def load_optimizer(self, fname):
        self.optimizer.load_state_dict(torch.load(fname, map_location=torch.device(torch.cuda.current_device())))

    def save_optimizer(self, fname):
        torch.save(self.optimizer.state_dict(), fname)

    def load_ckp(self,ckp):
        if self.opt_param == 'all':
            optimizer_use_model = os.path.join(self.train_output_dir,'weights',
                                               'Optimizer_{}.pth'.format(ckp))
        else:
            optimizer_use_model = os.path.join(self.train_output_dir,'weights', 'Optimizer_{}'.format(self.opt_param),
                                               'Optimizer_{}.h5'.format(ckp))

        self.load_optimizer(optimizer_use_model)

    def save_ckp(self,ckp):
        if self.opt_param=='all':
            optimizer_use_model = os.path.join(self.train_output_dir,'weights',
                                     'Optimizer_{}.pth'.format(ckp))
        else:
            os.makedirs(os.path.join(self.train_output_dir,'weights','Optimizer_{}'.format(self.opt_param)),exist_ok=True)

            optimizer_use_model = os.path.join(self.train_output_dir,'weights', 'Optimizer_{}'.format(self.opt_param),
                                               'Optimizer_{}.h5'.format(ckp))

        self.save_optimizer(optimizer_use_model)