import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F
from icons_dataset import IconsDataset
from torch.utils.data import DataLoader
import os

def mse_loss(pred,gt):
    return (((pred-gt)**2).sum(-1)).mean()

def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)

def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))

def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class ParallelModule(nn.Sequential):
    def __init__(self, *args):
        super(ParallelModule, self).__init__( *args )

    def forward(self, input):
        output = []
        for module in self:
            output.append( module(input) )
        return torch.cat( output, dim=1 )

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class ImageDownsampling(nn.Module):
    '''Generate samples in u,v plane according to downsampling blur kernel'''

    def __init__(self, sidelength, downsample=False):
        super().__init__()
        if isinstance(sidelength, int):
            self.sidelength = (sidelength, sidelength)
        else:
            self.sidelength = sidelength

        if self.sidelength is not None:
            self.sidelength = torch.Tensor(self.sidelength).cuda().float()
        else:
            assert downsample is False
        self.downsample = downsample

    def forward(self, coords):
        if self.downsample:
            return coords + self.forward_bilinear(coords)
        else:
            return coords

    def forward_box(self, coords):
        return 2 * (torch.rand_like(coords) - 0.5) / self.sidelength

    def forward_bilinear(self, coords):
        Y = torch.sqrt(torch.rand_like(coords)) - 1
        Z = 1 - torch.sqrt(torch.rand_like(coords))
        b = torch.rand_like(coords) < 0.5

        Q = (b * Y + ~b * Z) / self.sidelength
        return Q

class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output

class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations

class ModifiedFCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None, nodes_num = 100):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        outermost_layers = []
        for node in range(nodes_num):
            outermost_layers.append(MetaSequential(BatchLinear(hidden_features, out_features)))

        self.net.append(ParallelModule(*outermost_layers))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations

class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode

        self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
                                                    downsample=kwargs.get('downsample', False))
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(model_input, get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}

class MultipleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode

        self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
                                                    downsample=kwargs.get('downsample', False))
        self.net = ModifiedFCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(model_input, get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}

class JIRNet(MetaSequential):

    def __init__(self, images_size, images_num):
        super(JIRNet, self).__init__()

        self.images_size = images_size
        self.images_num = images_num

        self.net = MultipleBVPNet(out_features=3)

    def train(self):
        self.net.train()
        self.is_train = True

    def train_single_node(self,node_id):
        for n in range(self.images_num):
            requires_grad = False
            if n==node_id:
                requires_grad = True

            for param in self.net.net.net[-1][n].parameters():
                param.requires_grad = requires_grad

    def get_outermost_layers_weights(self):
        weights_matrix = np.zeros((self.images_num,256,3))
        for n in range(self.images_num):
            weights_matrix[n,:,:] = OrderedDict(self.net.net.net[-1][n][0].named_parameters())['weight'].clone().cpu().numpy().transpose((1,0))

        return weights_matrix

    def eval(self):
        self.net.eval()
        self.is_train = False

    def forward(self, u,v,n):
        BS = u.shape[0]
        # the input to the net is a point in the self.images_num+2 dimentional unit sphere
        input_vec = torch.zeros((BS,2), device=u.device)
        input_vec[:,0] = u.type(torch.float32)/self.images_size[1]
        input_vec[:,1] = v.type(torch.float32)/self.images_size[0]

        coeff_vec = torch.zeros((BS, self.images_num, 1), device=u.device)
        if isinstance(n,tuple):
            for bn in range(BS):
                coeff_vec[bn, n[0]] += n[2]
                coeff_vec[bn, n[1]] += 1 - n[2]
        else:
            for bn in range(BS):
                coeff_vec[bn,n[bn]] = 1.0

        outputs = self.net(input_vec.view(BS,1,2))

        output = (coeff_vec*outputs).sum(dim=1)

        return output

    def reconstruct_single_image(self, image_id):
        u, v, n = np.meshgrid(np.arange(0, self.images_size[1]), np.arange(0, self.images_size[0]), np.zeros(1) + image_id)
        u = u.reshape(-1).astype(np.int)
        v = v.reshape(-1).astype(np.int)
        n = n.reshape(-1).astype(np.int)

        u = torch.tensor(u).cuda()
        v = torch.tensor(v).cuda()
        n = torch.tensor(n).cuda()

        with torch.no_grad():
            I = torch.clamp(self.forward(u, v, n).view(self.images_size[0], self.images_size[1], 3),0,1)

        return I

    def upsample_single_image(self, image_id, factor):
        u, v, n = np.meshgrid(np.linspace(0, self.images_size[1], factor), np.linspace(0, self.images_size[0], factor), np.zeros(1) + image_id)

        H,W,_ = u.shape

        u = u.reshape(-1).astype(np.int)
        v = v.reshape(-1).astype(np.int)
        n = n.reshape(-1).astype(np.int)

        u = torch.tensor(u).cuda()
        v = torch.tensor(v).cuda()
        n = torch.tensor(n).cuda()

        with torch.no_grad():
            I = torch.clamp(self.forward(u, v, n).view(H, W, 3),0,1)

        return I

    def interpolate_image_pairs(self,image_id_0, image_id_1, alpha):
        u, v, n0 = np.meshgrid(np.arange(0, self.images_size[1]), np.arange(0, self.images_size[0]),
                              np.zeros(1) + image_id_0)
        u = u.reshape(-1).astype(np.int)
        v = v.reshape(-1).astype(np.int)
        n0 = n0.reshape(-1).astype(np.int)

        u = torch.tensor(u).cuda()
        v = torch.tensor(v).cuda()

        n0 = torch.tensor(n0).cuda()
        n1 = torch.zeros_like(n0) + image_id_1

        with torch.no_grad():
            I = torch.clamp(self.forward(u, v, (n0,n1,alpha)).view(self.images_size[0], self.images_size[1], 3), 0, 1)

        return I


    def run_step(self, batch, optimizer_wrapper):
        batch = [tensor.cuda() for tensor in batch]

        u, v, n, gt = batch
        output = self.forward(u, v, n)

        net_loss = mse_loss(output,gt)

        if self.is_train:
            optimizer_wrapper.optimizer.zero_grad()
            net_loss.backward()
            optimizer_wrapper.optimizer.step()

        return net_loss

    def load_ckp(self, ckp, train_output_dir):
        load_name = os.path.join(train_output_dir,'weights',
                                 'JIRNet_{}.pth'.format(ckp))
        self.net.load_state_dict(torch.load(load_name))

    def save_ckp(self, ckp, train_output_dir):
        save_name = os.path.join(train_output_dir,'weights',
                                 'JIRNet_{}.pth'.format(ckp))

        torch.save(self.net.state_dict(), save_name)

    def cuda(self):
        self.net.cuda()

if __name__ == '__main__':

    datapath = './data/48/'
    dataset = IconsDataset(datapath)

    dataloader = DataLoader(dataset, batch_size=5, num_workers=1, pin_memory=False, shuffle=False)

    net = JIRNet(dataset.images_size, dataset.images_num)
    net.cuda()

    for batch in dataloader:
        batch = [tensor.cuda() for tensor in batch]

        u, v, n, gt = batch
        output = net.forward(u,v,n)