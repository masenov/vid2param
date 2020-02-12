#import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir+'/PhysLoss')
sys.path.insert(0,parentdir+'/../models/PhysLoss')

from utils.viz import *
from utils.tools import *

import skvideo.io
import skvideo.datasets
import cv2


def generate_video_real(trajs, index=0, max_time=60, take_every_n_el=3, noise=False):
    count = 0
    print (index)
    bg, img, mask = generate_bg_real()
    bg = np.tile(np.expand_dims(bg, axis=0),(max_time//take_every_n_el,1,1,1))
    offset = 1 #np.random.randint(30)+
    blur_ratio = np.random.randint(15)+5
    padding = 200
    for i in range(0, max_time, take_every_n_el):
        x_pos, x_vel, y_pos, y_vel = trajs[index,i,:]
        x_pos/=10
        y_pos/=10
        ball_size = np.random.randint(10)+30#+np.random.randint(0)
        generate_image_real(bg[i//take_every_n_el], img, mask, x_pos=x_pos, y_pos=1-y_pos, x_vel=x_vel, y_vel=y_vel, ball_size=ball_size, offset=offset, blur_ratio=blur_ratio, noise=noise)
    return bg[:,padding:-padding,padding:-padding]


def generate_bg_real(bg=0, ball=1):
    bg = np.random.randint(550)
    ball = np.random.randint(700)
    ball_folder = np.random.randint(2)
    if (ball_folder):
        ball_folder = 'ping_pong'
    else:
        ball_folder = 'tennis_ball'
    ball_folder = 'ping_pong'
    bg_name='data/google_images/{}.png'.format(bg) # {:0>4d}.png
    img_name='data/'+ball_folder+'/{:d}.png'.format(ball)
    mask_name='data/'+ball_folder+'_mask/{:d}.png'.format(ball)
    bg_size=0.2
    padding=200 # 200


    bg = cv2.imread(bg_name)
    #bg = cv2.resize(bg, (0,0), fx=bg_size, fy=bg_size) # cv2.resize(oriimg,(int(newX),int(newY)))
    bg = cv2.resize(bg, (100,50)) # cv2.resize(oriimg,(int(newX),int(newY)))
    bg = np.pad(bg,[(padding,padding),(padding,padding),(0,0)],'constant',constant_values=(0,0))


    size = 50
    height, width = 2*(size+padding), 2*(size+padding)
    img = np.zeros((height, width, 3), np.uint8)
    mask = np.zeros((height, width, 3), np.float)
    row, col = size+padding,size+padding
    cv2.circle(img,(row, col), size, tuple ([int(x) for x in np.random.randint(255, size=3)]), -1)
    cv2.circle(mask,(row, col), size, (1.0,1.0,1.0), -1)


    """
    img = cv2.imread(img_name)
    mask = cv2.imread(mask_name)/255
    img[mask==0] = 0
    color_v = 0
    color = [color_v, color_v, color_v]
    img = cv2.copyMakeBorder(img,padding,padding,padding,padding,cv2.BORDER_CONSTANT, value=color)
    mask = cv2.copyMakeBorder(mask,padding,padding,padding,padding,cv2.BORDER_CONSTANT, value=color)
    """


    return bg, img, mask




def generate_image_real(bg, img, mask, x_pos=0.9, y_pos=0.1, x_vel=0, y_vel=0, ball_size=20, offset=30, blur_ratio=25, noise=False):

    padding=200


    angle = np.arctan2 (y_vel, x_vel)/(np.pi/180) + 90
    size = int(np.sqrt(y_vel*y_vel + x_vel*x_vel)*blur_ratio)
    if (size<3):
        size = 3


    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    image_center = tuple(np.array(kernel_motion_blur.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(kernel_motion_blur, rot_mat, kernel_motion_blur.shape[1::-1], flags=cv2.INTER_LINEAR)
    kernel_motion_blur = result

    # applying the kernel to the input image
    img_blur = cv2.filter2D(img, -1, kernel_motion_blur)
    mask_blur = cv2.filter2D(mask, -1, kernel_motion_blur)
    img_blur = cv2.resize(img_blur,(int(img_blur.shape[0]/ball_size),int(img_blur.shape[1]/ball_size)))
    mask_blur = cv2.resize(mask_blur,(int(mask_blur.shape[0]/ball_size),int(mask_blur.shape[1]/ball_size)))


    if (noise):
        x_pos += (np.random.rand()-0.5)/30
        y_pos += (np.random.rand()-0.5)/30
    x_pos = int((bg.shape[1]-2*padding)*x_pos+padding-img_blur.shape[0]/2)
    y_pos = int((bg.shape[0]-2*padding)*y_pos+padding-img_blur.shape[1]/2-offset)
    bg[y_pos:(y_pos+img_blur.shape[0]),x_pos:(x_pos+img_blur.shape[1])] = img_blur+(1-mask_blur)*bg[y_pos:(y_pos+img_blur.shape[0]),x_pos:(x_pos+img_blur.shape[1])]
    #cv2.imshow('Motion Blur', bg)
    #cv2.waitKey(0)
    #new_image=new_image[padding:-padding,padding:-padding]
    #print (bg.shape)
    #return new_image




if __name__ == "__main__":
    data = np.load('data/data.npz')
    trajs = data['arr_1'] # 100x200x4 hor pos, ver pos, hor vel, ver vel
    #dyn = np.swapaxes(np.swapaxes(trajs,0,2),1,2)
    """
    params = torch.from_numpy(data['arr_2'])
    params_intervals = data['arr_3'].item(0).items()
    norm_trajs, params = normalize_all_data(data, dyn, params, params_intervals)
    data = np.load('data/data.npz')
    trajs = data['arr_1'] # 100x200x4 hor pos, ver pos, hor vel, ver vel
    dyn = np.swapaxes(np.swapaxes(trajs,0,2),1,2)
    bg = cv2.resize(cv2.imread('data/frames_lab/0001.jpg'),(400,400))
    """
    video = generate_video_real(trajs)
    skvideo.io.vwrite("outputvideo.mp4", video.astype(np.uint8))


