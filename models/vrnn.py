import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
import numpy as np
import timeit

# STN module # https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

class STN(nn.Module):
    def __init__(self, input_dims=1, dataset="mnist"):
        super(STN, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(input_dims, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        if (dataset=="mnist"):
            self.fc_loc_inp_dim = 10 * 3 * 3
        else:
            self.fc_loc_inp_dim = 10 * 50 * 92
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_loc_inp_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_loc_inp_dim)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)


        size_x = torch.empty(4, dtype=torch.int)
        size_x[0] = x.size()[0]
        size_x[1] = x.size()[1]
        size_x[2] = x.size()[2]//8
        size_x[3] = x.size()[3]//8
        grid = F.affine_grid(theta, size_x)
        x = F.grid_sample(x, grid)

        return x, theta

    def forward(self, x):
        # transform the input
        x, theta = self.stn(x)

        return x, theta

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for
inference, prior, and generating models."""

class decoder_cnn(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, dataset='mnist', input_dim=62):
        super(decoder_cnn, self).__init__()
        self.dataset = dataset
        if dataset == 'mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = input_dim
            self.output_dim = 1
        else:
            self.input_height = 100
            self.input_width = 200
            self.input_dim = input_dim
            self.output_dim = 3


        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 32 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(32 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 32, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)
        if (self.dataset=="mnist"):
            x = x.view(-1,784)
        else:
            x = x.view(-1,100*200*3)
        return x

class encoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset = 'mnist', z_dim=62, spectral=False):
        super(encoder, self).__init__()
        self.z_dim = z_dim
        self.dataset = dataset
        if dataset == 'mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1
            self.fc = 16 * 7 * 7
        elif dataset == 'mnist_real':
            self.input_height = 25 #50
            self.input_width = 50 #100
            self.input_dim = 1
            self.output_dim = 1
            self.fc = 16 * 12 * 25 #1152
        else:
            self.input_height = 100
            self.input_width = 200
            self.input_dim = 3
            self.output_dim = 3
            self.fc = 16 * 12 * 25
            self.fc = 20000


        if (spectral):
            self.conv = nn.Sequential(
                SpectralNorm(nn.Conv2d(self.input_dim, 16, 3, 2, 2)),
                nn.LeakyReLU(0.2),
                SpectralNorm(nn.Conv2d(16, 16, 3, 2, 0)),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2),
            )
            self.fc = nn.Sequential(
                #nn.Linear(32 * (self.input_height // 4) * (self.input_width // 4), 64),
                SpectralNorm(nn.Linear(self.fc, 256)),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, self.z_dim),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(self.input_dim, 16, 3, 2, 2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(16, 16, 3, 2, 0),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2),
            )
            self.fc = nn.Sequential(
                #nn.Linear(32 * (self.input_height // 4) * (self.input_width // 4), 64),
                nn.Linear(self.fc, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, self.z_dim),
            )


    def forward(self, input):
        if (self.dataset=="mnist"):
            input = input.view(-1,1,28,28)
        elif (self.dataset=="mnist_real"):
            input = input.view(-1,1,50,100) #25, 50 # 216,384)
        else:
            input = input.view(-1,3,100,200) # 216,384)
        x = self.conv(input)
        #x = x.view(-1, 32 * (self.input_height // 4) * (self.input_width // 4))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, device, beta=1, alpha=1, gamma=1, theta=1, bias=False, original_paper=False, dataset="mnist", decoder=False, spectral=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.original_paper = original_paper
        self.dataset = dataset
        self.decoder = decoder
        self.count = 0

        """
        if (self.dataset=="mnist"):
            self.stn_tracker = STN(input_dims=1, dataset="mnist")
        else:
            self.stn_tracker = STN(input_dims=3, dataset="real")
        """
        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_x = encoder(z_dim=h_dim, dataset=dataset, spectral=spectral)


        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())



        self.phi_h = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        #encoder
        enc_dec_dim = 3*h_dim
        prior_dim = 2*h_dim
        if (spectral):
            self.enc = nn.Sequential(
                SpectralNorm(nn.Linear(enc_dec_dim, h_dim)),
                nn.ReLU(),
                SpectralNorm(nn.Linear(h_dim, h_dim)),
                nn.ReLU())
            self.enc_mean = SpectralNorm(nn.Linear(h_dim, z_dim))
            self.enc_std = nn.Sequential(
                SpectralNorm(nn.Linear(h_dim, z_dim)),
                nn.Softplus())

            #prior
            self.prior = nn.Sequential(
                SpectralNorm(nn.Linear(prior_dim, h_dim)),
                nn.ReLU())
            self.prior_mean = SpectralNorm(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(
                SpectralNorm(nn.Linear(h_dim, z_dim)),
                nn.Softplus())
        else:
            self.enc = nn.Sequential(
                nn.Linear(enc_dec_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU())
            self.enc_mean = nn.Linear(h_dim, z_dim)
            self.enc_std = nn.Sequential(
                nn.Linear(h_dim, z_dim),
                nn.Softplus())

            #prior
            self.prior = nn.Sequential(
                nn.Linear(prior_dim, h_dim),
                nn.ReLU())
            self.prior_mean = nn.Linear(h_dim, z_dim)
            self.prior_std = nn.Sequential(
                nn.Linear(h_dim, z_dim),
                nn.Softplus())


        #decoder
        if (decoder):
            self.dec_mean = decoder_cnn(input_dim=h_dim, dataset=dataset)
            self.dec = nn.Sequential(
                nn.Linear(enc_dec_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU())
            self.dec_std = nn.Sequential(
                nn.Linear(h_dim, x_dim),
                nn.Softplus())
            self.dec_mean = nn.Sequential(
                nn.Linear(h_dim, x_dim),
                nn.Sigmoid())



        #recurrence
        if (decoder):
            self.rnn = nn.LSTM(h_dim, h_dim, n_layers, bias)
        else:
            self.rnn = nn.LSTM(h_dim, h_dim, n_layers, bias)

        # device to train on
        self.device = device

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    # TODO: Need to be fixed with cnn encoders and other changes I have made
    def reset(self):
        #feature-extracting transformations
        self.phi_x = nn.Sequential(
                nn.Linear(self.x_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())

        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU())







        #encoder
        self.enc = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(self.h_dim, self.x_dim),
            nn.Softplus())
        self.dec_mean = nn.Sequential(
                nn.Linear(self.h_dim, self.x_dim),
                nn.Sigmoid())





    def forward(self, x, h=None, pred=None, dyn=None, params=None, epoch=0, sw=None):
        all_enc_mean, all_enc_std = torch.zeros(x.size(0),x.size(1),self.z_dim).to(self.device), torch.zeros(x.size(0),x.size(1),self.z_dim).to(self.device)
        all_dec_mean, all_dec_std = torch.zeros(x.shape).to(self.device), torch.zeros(x.shape).to(self.device)
        dec_std_t = 1
        dec_mean_t = 1
        all_z_t_mean = torch.zeros(x.size(0),x.size(1),self.z_dim).to(self.device)
        all_z_t_var = torch.zeros(x.size(0),x.size(1),self.z_dim).to(self.device)
        all_z_t = torch.zeros(x.size(0),x.size(1),self.z_dim).to(self.device)
        kld_loss = 0
        nll_loss = 0
        phys_loss = 0
        if h is None:
            h1 = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim)).to(self.device)
            c1 = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim)).to(self.device)
        else:
            h1 = h[0]
            c1 = h[1]
            x_t_1 = pred

        start = 0
        z_prev = None
        z_current = None
        for t in range(start, x.size(1)):
            if (pred is None):
                phi_x_t = self.phi_x(x[:,t])


                #encoder
                enc_t = self.enc(torch.cat([phi_x_t, h1[-1], c1[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                #enc_mean_t = torch.clamp(enc_mean_t, max=1.0)
                enc_std_t = self.enc_std(enc_t)
                #enc_std_t = torch.clamp(enc_std_t, max=1.0)

            #prior
            prior_t = self.prior(torch.cat([h1[-1],c1[-1]],1))
            prior_mean_t = self.prior_mean(prior_t)
            #prior_mean_t = torch.clamp(prior_mean_t, max=1.0)
            prior_std_t = self.prior_std(prior_t)
            #prior_std_t = torch.clamp(prior_std_t, max=1.0)

            #sampling and reparameterization
            if (pred is None):
                z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            else:
                z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)

            all_z_t_mean[:,t,:] = prior_mean_t
            all_z_t_var[:,t,:] = prior_std_t
            all_z_t[:,t,:] = z_t

            phi_z_t = self.phi_z(z_t)


            #decoder
            if (self.decoder):
                dec_t = self.dec(torch.cat([phi_z_t, h1[-1], c1[-1]], 1))
                #dec_mean_t = torch.clamp(self.dec_mean(dec_t),0.0,1.0)
                dec_mean_t = self.dec_mean(dec_t)
                if (pred is not None):
                    phi_x_t = self.phi_x(dec_mean_t)
                #_, (h1, c1) = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h1, c1))
                _, (h1, c1) = self.rnn(phi_z_t.unsqueeze(0), (h1, c1))
            else:
                _, (h1, c1) = self.rnn(phi_z_t.unsqueeze(0), (h1, c1))


            if (pred is None):
                #computing losses
                kld_loss += self.beta * self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
                if (self.decoder):
                    nll_loss += self.gamma * F.binary_cross_entropy(dec_mean_t, x[:,t], reduction='sum')
                else:
                    nll_loss = 0
                if dyn is not None:
                    pos_x = torch.unsqueeze(dyn[0][:,t],1)
                    pos_y = torch.unsqueeze(dyn[1][:,t],1)
                    rest = torch.unsqueeze(params[:,2],1)
                    grav = torch.unsqueeze(params[:,3],1)
                    drag = torch.unsqueeze(params[:,4],1)
                    roll = torch.unsqueeze(params[:,5],1)
                    all_t = torch.cat((pos_x,pos_y,rest,grav,drag,roll),1).float().to(self.device)
                    phys_loss += self.alpha * self._nll_gauss(enc_mean_t[:,:6], enc_std_t[:,:6], all_t[:,:6])

                all_enc_std[:,t] = enc_std_t
                all_enc_mean[:,t] = enc_mean_t
            if (self.decoder):
                x_t_1 = dec_mean_t
                all_dec_mean[:,t] = dec_mean_t
                all_dec_std[:,t] = dec_std_t
        h = [h1, c1]
        return kld_loss, nll_loss, phys_loss,  \
            torch.cat((torch.unsqueeze(all_enc_mean,0),torch.unsqueeze(all_enc_std,0)),0), \
            torch.cat((torch.unsqueeze(all_dec_mean,0),torch.unsqueeze(all_dec_std,0)),0), \
            h, dec_mean_t, all_z_t_mean, all_z_t_var, all_z_t

    def forward_eval(self, x, get_h=None, h=None, max_time=0, pred=False):
        if (h is None):
            all_z_t_mean = torch.zeros(x.size(0),x.size(1),self.z_dim).to(self.device)
            all_z_t_std  = torch.zeros(x.size(0),x.size(1),self.z_dim).to(self.device)
            h1 = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim)).to(self.device)
            c1 = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim)).to(self.device)
            max_time = x.size(1)
        else:
            h1 = h[0]
            c1 = h[1]
            all_z_t_mean = torch.zeros(x.size(0),max_time,self.z_dim).to(self.device)
            all_z_t_std  = torch.zeros(x.size(0),max_time,self.z_dim).to(self.device)

        time_eval = np.zeros(max_time)
        for t in range(0, max_time):
            if (pred):
                start_time = timeit.default_timer()
                prior_t = self.prior(torch.cat([h1[-1],c1[-1]],1))
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)

                z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)

                phi_z_t = self.phi_z(z_t)


                _, (h1, c1) = self.rnn(phi_z_t.unsqueeze(0), (h1, c1))
                all_z_t_mean[:,t,:] = prior_mean_t
                all_z_t_std[:,t,:]  = prior_std_t
                time_eval[t] = time_eval[max(0,t-1)] + timeit.default_timer() - start_time
            else:
                start_time = timeit.default_timer()
                phi_x_t = self.phi_x(x[:,t])


                enc_t = self.enc(torch.cat([phi_x_t, h1[-1], c1[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)

                z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)

                phi_z_t = self.phi_z(z_t)


                _, (h1, c1) = self.rnn(phi_z_t.unsqueeze(0), (h1, c1))
                all_z_t_mean[:,t,:] = enc_mean_t
                all_z_t_std[:,t,:] = enc_std_t
                time_eval[t] = time_eval[max(0,t-1)] + timeit.default_timer() - start_time


        if (get_h):
            h = [h1, c1]
            return all_z_t_mean, all_z_t_std, time_eval, h
        else:
            return all_z_t_mean, all_z_t_std, time_eval


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(self.device)
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        #std_1 = torch.exp(0.5*std_1)
        #std_2 = torch.exp(0.5*std_2)

        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)
        return  0.5 * torch.sum(kld_element)



    def _nll_gauss(self, mean, std, x):
        neg_log_p = ((mean - x) ** 2 / (2 * torch.pow(std,2))) + torch.log(torch.pow(std,2))/2 + torch.log(torch.from_numpy(np.array(2*np.pi)))/2
        return neg_log_p.sum()







