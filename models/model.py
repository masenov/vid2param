from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from tensorboardX import SummaryWriter

from physicsdata.bouncing_ball import *
from datasets.bouncing_ball import ode_ball



class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(4, 30)
        self.lstm2 = nn.LSTMCell(30, 30)
        self.linear = nn.Linear(30, 4)

    def forward(self, input, future = 0):
        device = torch.device("cuda:0")
        outputs = torch.zeros(input.size(0), input.size(1) + future, 4, dtype=torch.double).to(device)
        h_t = torch.zeros(input.size(0), 30, dtype=torch.double).to(device)
        c_t = torch.zeros(input.size(0), 30, dtype=torch.double).to(device)
        h_t2 = torch.zeros(input.size(0), 30, dtype=torch.double).to(device)
        c_t2 = torch.zeros(input.size(0), 30, dtype=torch.double).to(device)
        input_timesteps = input.size(1)
        import pdb; pdb.set_trace()
        for i, input_t in enumerate(input.chunk(input_timesteps, dim=1)):
            h_t, c_t = self.lstm1(input_t[:,0,:4], (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            """
            for j in range(train_future):
                if (j==0):
                    temp_input_1 = output[:,0] + input_t[:,0,0]
                else:
                    temp_input_1 = output[:,0] + temp_input_1
                temp_input_2 = output[:,0]
                temp_input_3 = output[:,1]
                h_t, c_t = self.lstm1(torch.cat((temp_input_1,temp_input_2,temp_input_3)).view(-1,3), (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                output = self.linear(h_t2)
            """
            outputs[:,i,:] = output[:,:]
            #outputs[:,i,0] = output[:,0] + input_t[:,0,0]
            #outputs[:,i,1] = output[:,0]
            #outputs[:,i,2] = output[:,1]
        for i in range(future):# if we should predict the future
            """
            if (i==0):
                pos = output[:,0] + input_t[:,0,0]
            else:
                pos = output[:,0] + pos
            acc = input_t[:,0,2]
            res = input_t[:,0,3]
            current_data = torch.cat((pos,output[:,0],acc,res)).view(4,len(pos)).permute(1,0)
            h_t, c_t = self.lstm1(current_data, (h_t, c_t))
            """
            h_t, c_t = self.lstm1(output[:,:], (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            #output = (torch.sin((output[:,0])/20.0)).view(-1,1)
            outputs[:,i+input_timesteps,:] = output[:,:]
            #outputs[:,i+input_timesteps,0] = output[:,0] + outputs[:,i+input_timesteps-1,0]
            #outputs[:,i+input_timesteps,1] = output[:,0]
            #outputs[:,i+input_timesteps,2] = output[:,1]
        #outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


