import os
import sys
sys.path.append(os.getcwd())

sys.path.insert(0, '/home/martin/DL/PhysVideo/')
sys.path.insert(0, '/data/DL/PhysVideo/')
import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms, datasets
import numpy as np
import cv2
from datasets.bouncing_ball import ode_ball

"""
def bouncing_ball_images(folder="datasets/bouncing_ball/BouncingBall"):
    images = np.zeros((100,500,64,64))
    for i in range(100):
        for j in range(500):
            im = (cv2.imread(folder+"/"+str(i)+"/"+str(j)+".png"))
            #im = (im+np.random.random_sample(im.shape))/255.0-0.5
            im = (im+np.random.random_sample(im.shape))/255.0
            im = cv2.resize(im, (64, 64))
            #im = cv2.resize((cv2.imread(folder+"/"+str(i)+"/"+str(j)+".png"))/255.0, (25, 25))
            images[i,j,:,:] = im[:,:,0]
            print (i, j)
    return images


class BouncingBall(Dataset):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ntotal = 500
        nsample = 500
        self.orig_trajs = bouncing_ball_images()
        self.orig_trajs = torch.from_numpy(self.orig_trajs.reshape(100,ntotal,4096) + np.random.rand(100,ntotal,4096)/100).float()
        #np.take(self.orig_trajs,np.random.rand(self.orig_trajs.shape[0]).argsort(),axis=0,out=self.orig_trajs)
        self.samp_trajs = self.orig_trajs[:,::ntotal//nsample,:]
        self.orig_ts = np.linspace(0,4,500)
        self.samp_ts = self.orig_ts[::ntotal//nsample]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def __getitem__(self, index):
        input = self.orig_trajs[index,:250,:].view(250,1,64,64)
        output = self.orig_trajs[index,250:,:].view(250,1,64,64)
        return input, output

    def __len__(self):
        return 50
"""


def bouncing_ball_images(folder="data/dataset3", nimages=200, ntimesteps=200, size=28):
    images = np.zeros((nimages,ntimesteps,size,size,3))
    # Number of images
    for i in range(nimages):
        # Number of timesteps
        for j in range(ntimesteps):
            im = (cv2.imread(folder+"/video_"+str(i)+"_"+str(j)+".png"))
            im = cv2.resize(im, (size, size))
            im = im/255.0
            images[i,j,:,:,:] = im
            print (i, j)
    images = np.swapaxes(np.swapaxes(images,4,2),3,4)
    return images


class BouncingBall(Dataset):
    def __init__(self, gpu=0):
        self.to_tensor = transforms.ToTensor()
        self.ntotal = 10
        self.nsample = 200
        self.ntotal = 300
        self.size = 28
        self.frames_input = 100
        self.frames_output = 100
        self.orig_trajs = bouncing_ball_images(nimages=self.ntotal,ntimesteps=self.nsample,size=self.size)
        self.orig_trajs = torch.from_numpy(self.orig_trajs).float()
        self.data = np.load('data/train3.npz')
        self.trajs = torch.from_numpy(self.data['arr_0'])[:self.ntotal,2:,:]
        self.params = torch.from_numpy(self.data['arr_1'])[:self.ntotal]
        self.params_intervals = self.data['arr_2']
        self.device = gpu


    def __getitem__(self, index):
        input = self.orig_trajs[index,:self.frames_input,:1]
        output = self.orig_trajs[index,self.frames_input:,:1]
        x, vel_x, y, vel = self.trajs[index].t()
        params = self.params[index]
        time = torch.from_numpy(np.linspace(0,20,self.frames_input+self.frames_output))
        return input, output, x, vel_x, y, vel, time, params

    def __len__(self):
        return self.ntotal




class BouncingBallNumpy(Dataset):
    def __init__(self, frames_input, frames_output, image_size,gpu):
        self.to_tensor = transforms.ToTensor()
        self.device = gpu
        self.frames_input = frames_input
        self.frames_output = frames_output
        self.image_size = image_size


    def __getitem__(self, index):
        #images = ode_ball.create_images(frames_input=self.frames_input,frames_output=self.frames_output,reshape=self.image_size)
        images, x, vel_x, y, vel, time, params  = ode_ball.create_images(frames_input=self.frames_input, frames_output=self.frames_output, reshape=self.image_size)
        input = torch.from_numpy(images[:self.frames_input,:,:]).view(self.frames_input,1,self.image_size,self.image_size).float()
        output = torch.from_numpy(images[self.frames_input:,:,:]).view(self.frames_output,1,self.image_size,self.image_size).float()
        return input, output, x, vel_x, y, vel, time, params

    def __len__(self):
        return 512


class BouncingBallPreloaded(Dataset):
    def __init__(self, frames_input, frames_output, image_size_x, image_size_y, color_channels, gpu, name="train_data_big", changing_physics=0):
        self.to_tensor = transforms.ToTensor()
        self.device = gpu
        self.frames_input = frames_input
        self.frames_output = frames_output
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.color_channels = color_channels
        #self.data = torch.from_numpy(np.load)
        try:
            self.data = np.load('data/'+name+'.npz')
        except:
            self.data = np.load(name+'.npz')
        self.images = torch.from_numpy(self.data['arr_0'])
        self.trajs = torch.from_numpy(self.data['arr_1'])
        self.params = torch.from_numpy(self.data['arr_2'])
        self.params_intervals = self.data['arr_3']
        #self.multiplier = 200//(frames_input+frames_output)
        self.multiplier = 1
        self.changing_physics = changing_physics



    def __getitem__(self, index):
        images = self.images[index]
        # MNIST
        if len(images.shape)==3:
            input = images[:self.frames_input*self.multiplier:self.multiplier].view(self.frames_input,self.color_channels,self.image_size_x,self.image_size_y)
            output = images[self.frames_input*self.multiplier:(self.frames_input+self.frames_output)*self.multiplier:self.multiplier].view(self.frames_output,self.color_channels,self.image_size_x,self.image_size_y)
        # REAL
        else:
            input = images[:self.frames_input*self.multiplier:self.multiplier].permute(0,3,1,2)
            output = images[self.frames_input*self.multiplier:(self.frames_input+self.frames_output)*self.multiplier:self.multiplier].permute(0,3,1,2)
        x, vel_x, y, vel = self.trajs[index,::self.multiplier].t()
        if (self.changing_physics):
            params = self.params[index*self.changing_physics:(index+1)*self.changing_physics,:]
        else:
            params = self.params[index,:]
        time = torch.from_numpy(np.linspace(0,10,self.frames_input+self.frames_output))
        return input, output, x, vel_x, y, vel, time, params

    def __len__(self):
        return self.images.shape[0]




class BouncingBallRandomPosition(Dataset):
    def __init__(self, image_size):
        self.to_tensor = transforms.ToTensor()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size


    def __getitem__(self, index, radius=60, original=256, reshape=28):
        x, y = np.random.uniform(radius, original-radius, 2)
        image = ode_ball.create_image(original=original, x=x, y=y, radius=30, reshape=28)
        image = np.reshape(image, (1,reshape,reshape))
        image -= image.min()
        image /= image.max()
        return image, [x,y]

    def __len__(self):
        return 640







if __name__ == "__main__":
    dataset = BouncingBallNumpy(10,10,28)
    dataset.__getitem__(0)
