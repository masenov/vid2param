import sys, os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import torchvision
import torch.optim as optim
from tensorboardX import SummaryWriter
import argparse
from models.vrnn import *
from utils.viz import *
from utils.tools import *
import matplotlib
from collections import OrderedDict
matplotlib.use('Agg')


parser = argparse.ArgumentParser()

# Number of GPU to use
parser.add_argument('--gpu', type=int, default=0)
# Number of iterations to train
parser.add_argument('--niters', type=int, default=2000)
# Experiment namer
parser.add_argument('--exp_name', type=str, default='demo_exp')
# Batch size
parser.add_argument('--batch_size', type=int, default=16)
# Number of frames before prediction
parser.add_argument('--back_frames', type=int, default=100)
# Number of prediction frames
parser.add_argument('--forw_frames', type=int, default=100)
# Beta for the KL divergence
parser.add_argument('--beta', type=int, default=1)
# Alpha for the Phys
parser.add_argument('--alpha', type=int, default=10)
# Gamma for the reconstruction
parser.add_argument('--gamma', type=int, default=1)
# Theta for the same z encoding
parser.add_argument('--theta', type=int, default=1)
# Size of the latent space
parser.add_argument('--z_dim', type=int, default=30)
# Size of the hidden space
parser.add_argument('--h_dim', type=int, default=400)
# Original paper - feeding the image into the recurrent bit?
parser.add_argument('--original_paper', type=int, default=0)
# Dataset name - name of the dataset used
parser.add_argument('--dataset', type=str, default="dataset")
# Train or eval?
parser.add_argument('--train', type=int, default=1)
# Changing physics - randomize the data so the physics are changing - 4
parser.add_argument('--changing_physics', type=int, default=0)
# B&W videos or the real ones
parser.add_argument('--dataset_type', type=str, default="mnist")
# Decoder or not
parser.add_argument('--decoder', type=int, default=1)
# ICRA percentage
parser.add_argument('--ICRA_percentage', type=float, default=0)

args = parser.parse_args()

sw = SummaryWriter('runs/'+args.exp_name)

import ode_ball
from bouncing_ball import *


batch_size = args.batch_size

device = torch.device('cuda:' + str(args.gpu))

if (args.dataset_type=="mnist"):
    train_dset = BouncingBallPreloaded(args.back_frames,args.forw_frames,28,28,1,args.gpu,name="train_" + args.dataset,changing_physics=args.changing_physics)
elif (args.dataset_type=="mnist_real"):
    train_dset = BouncingBallPreloaded(args.back_frames,args.forw_frames,25,50,1,args.gpu,name="train_" + args.dataset,changing_physics=args.changing_physics)
else:
    train_dset = BouncingBallPreloaded(args.back_frames,args.forw_frames,100,200,3,args.gpu,name="train_" + args.dataset,changing_physics=args.changing_physics)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=0)

params_intervals = train_dset.params_intervals




#params_intervals = OrderedDict([('horizontal_position', [0, 10]), ('horizontal_velocity', [-8, -4, 4, 8]), ('vertical_position', [6, 8]), ('vertical_velocity', [-4, -3, 3, 4]), ('rest', [0.6, 1.0]), ('gravity', [6.81, 12.81]), ('drag_coefficient', [0.05, 0.0005]), ('rolling_coefficient', [0.0, 0.7])]).items()

if (args.dataset_type=="mnist"):
    test_dset = BouncingBallPreloaded(args.back_frames,args.forw_frames,28,28,1,args.gpu,name="test_" + args.dataset,changing_physics=args.changing_physics)
elif (args.dataset_type=="mnist_real"):
    test_dset = BouncingBallPreloaded(args.back_frames,args.forw_frames,25,50,1,args.gpu,name="test_" + args.dataset,changing_physics=args.changing_physics)
else:
    test_dset = BouncingBallPreloaded(args.back_frames,args.forw_frames,100,200,3,args.gpu,name="test_" + args.dataset,changing_physics=args.changing_physics)
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=True, num_workers=0)

if (args.dataset_type=="mnist"):
    model = VRNN(x_dim=784, h_dim=args.h_dim, z_dim=args.z_dim, n_layers=2, alpha=args.alpha, beta=args.beta, gamma=args.gamma, device=device, original_paper=args.original_paper, dataset=args.dataset_type, decoder=args.decoder).to(device)
elif (args.dataset_type=="mnist_real"):
    model = VRNN(x_dim=50*100*1, h_dim=args.h_dim, z_dim=args.z_dim, n_layers=2, alpha=args.alpha, beta=args.beta, gamma=args.gamma, device=device, original_paper=args.original_paper, dataset=args.dataset_type, decoder=args.decoder).to(device)
else:
    model = VRNN(x_dim=100*200*3, h_dim=args.h_dim, z_dim=args.z_dim, n_layers=2, alpha=args.alpha, beta=args.beta, gamma=args.gamma, device=device, original_paper=args.original_paper, dataset=args.dataset_type, decoder=args.decoder).to(device)

if (not args.train):
    model.load_state_dict(torch.load('trained_weights',map_location='cuda:0'))

optimizer = optim.Adam(model.parameters(), lr=1e-3)


train_loss = 0
itr = 0
for epoch in range(1, args.niters):
    for batch_idx, (input, output, x, vel_x, y, vel, time, params) in enumerate(train_loader):
        model.train()
        model.count = epoch
        data = torch.cat((input,output),1)
        data = data.to(device).float()
        dyn = [x,vel_x,y,vel]
        dyn, params = normalize_all_data(data, dyn, params, params_intervals)
        params = add_noise(args.ICRA_percentage,params,params_intervals)
        if (batch_idx == 0 and epoch == 1):
            if (args.dataset_type=="mnist"):
                sw.add_video('data',data[:min(args.batch_size,8)],epoch,fps=30)
            else:
                sw.add_video('data',data[:min(args.batch_size,8)],epoch,fps=30)
            viz_data_tb(x,vel_x,y,vel,params,sw)
        if (args.train):
            optimizer.zero_grad()
            kld_loss, nll_loss, phys_loss, _, dec, h, _, _, _, _ = model(data.view(data.shape[0],args.back_frames+args.forw_frames,-1),dyn=dyn,params=params, epoch=epoch, sw=sw)
            if (args.decoder):
                loss = kld_loss + phys_loss + nll_loss
            else:
                loss = kld_loss + phys_loss
            current_batch_size = input.shape[0]
            sw.add_scalar('loss', loss/current_batch_size, itr)
            sw.add_scalar('kld_loss', kld_loss/args.beta/current_batch_size, itr)
            sw.add_scalar('nll_loss', nll_loss/args.gamma/current_batch_size, itr)
            sw.add_scalar('phys_loss', phys_loss/args.alpha/current_batch_size, itr)
            itr += 1
            loss.backward()
            optimizer.step()
        if epoch%5 == 0 and batch_idx == 0:
            acc_latent_space(test_loader, 'Test', model, device, params_intervals, args, sw, epoch, args.back_frames+args.forw_frames)
            test_pred_model(test_loader, 'Test', model, device, params_intervals, args, sw, epoch, args.decoder)
            torch.save(model.state_dict(),'weights/'+args.exp_name+"_"+str(epoch))
            if (not args.train):
                sw.close()
                sys.exit()
sw.close()
