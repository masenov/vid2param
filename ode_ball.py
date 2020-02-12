# ODE solver for generating bouncing ball trajectories and videos
# Due to the efficient solver and parallel implementation, it is
# able to generated thousdands of accurate trajectories within a few minutes

# For more details on ODE solvers refer to information here: https://uk.mathworks.com/help/matlab/ordinary-differential-equations.html

from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from scipy.integrate import odeint, RK45, solve_ivp
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
import timeit
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir+'/../datasets/bouncing_ball/')
from collections import OrderedDict


from blur import *
import skvideo.io
import skvideo.datasets




#np.random.seed(0)

def create_image(original, x, y, radius, reshape):
    img = np.ones((original, original))
    center = [x, y]
    radius = radius
    mask = create_circular_mask(original, original, center=center, radius=radius)
    img[~mask] = 0.0
    img = cv2.resize(img, dsize=(reshape, reshape), interpolation=cv2.INTER_LINEAR)
    #img[img<0.3] = 0.0
    return img

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask




def func(t, X, P):
    dx0 = X[1]
    dy0 = X[3]
    speed = np.sqrt(dx0**2 + dy0**2)
    drag_x = P[2]*dx0*speed/1.0
    drag_y = P[2]*dy0*speed/1.0
    dx1 = 0
    dy1 = -P[1]
    dx1 = -drag_x
    dy1 = -P[1] - drag_y
    return [dx0, dx1, dy0, dy1]

def func_roll(t, X, P):
    dx0 = X[1]
    dy0 = 0
    """
    if (np.abs(X[1])<0.000001):
        dx1 = 0
    else:
    """
    dx1 = -1*np.sign(X[1])*P[3]
    #dx1 = P[2]*P[3]
    dy1 = 0
    return [dx0, dx1, dy0, dy1]




def f(t, X, P):
    return func(t, X, P)

def f2(X, t, P):
    return func(t, X, P)

def f_roll(t, X, P):
    return func_roll(t, X, P)

def f2_roll(X, t, P):
    return func_roll(t, X, P)



def calculate_IC(impactState, r):
    IC = np.array([impactState[0],impactState[1],impactState[2],impactState[3]])
    if (np.abs(impactState[0]-10)<0.01 and impactState[1]>0):
        IC = np.array([IC[0],-1*IC[1],IC[2],IC[3]])
    if (np.abs(impactState[0])<0.01 and impactState[1]<0):
        IC = np.array([IC[0],-1*IC[1],IC[2],IC[3]])
    if (np.abs(impactState[2])<0.01 and impactState[3]<0):
        IC = np.array([IC[0],IC[1],IC[2],-r*IC[3]])
    return IC


# horizontal position, horizontal velocity, vertical position, vertical velocity
def hit_ground(t, y):
    # Hit right wall
    if (y[0]>10 and y[1]>0):
        # negative number
        return -1
    # Hit left wall
    if (y[0]<0 and y[1]<0):
        # negative number
        return -1
    # Vertical velocity is negative and we hit below 0
    if (y[2]<0 and y[3]<0):
        return -1
    return 1
hit_ground.terminal = True
hit_ground.direction = 0

def hit_ground_roll(t, y):
    # Hit right wall
    if (y[0]>10 and y[1]>0):
        # negative number
        return -1
    # Hit left wall
    if (y[0]<0 and y[1]<0):
        # negative number
        return -1
    return 1
hit_ground_roll.terminal = True
hit_ground_roll.direction = 0







def interval(l, n):
    if (len(l)==2):
        w = (l[1] - l[0]) / n
        intervals = np.array([np.array([l[0]+i*w, l[0]+(i+1)*w]) for i in range(n)])
    elif (len(l)==4):
        w1 = (l[1] - l[0]) / (n//2)
        intervals1 = np.array([np.array([l[0]+i*w1, l[0]+(i+1)*w1]) for i in range(n//2)])
        w2 = (l[3] - l[2]) / (n//2)
        intervals2 = np.array([np.array([l[2]+i*w2, l[2]+(i+1)*w2]) for i in range(n//2)])
        intervals = np.concatenate((intervals1,intervals2))
    return intervals

def generate_parameters(Physics_Params, n, interpolation_intervals, train):
    parameters = np.zeros((n, len(Physics_Params)))
    for i, (name, phys_interval) in enumerate(Physics_Params):
        if (interpolation_intervals==1):
            if (len(interval)==2):
                parameter = np.random.uniform(*phys_interval,size=n)
            else:
                param1 = np.random.uniform(*phys_interval[:2],size=n//2)
                param2 = np.random.uniform(*phys_interval[2:],size=n//2)
                parameter = np.random.shuffle(np.hstack((param1,param2)))
        else:
            intervals = interval(phys_interval, interpolation_intervals)
            inter = np.random.randint(interpolation_intervals//2, size=n)*2
            if (train):
                inter+=1
            parameter = np.random.uniform(*intervals[inter].T)
        parameters[:, i] = parameter
    return parameters

def generate_traj(IC, P, max_bounce, max_time, n_data, time_length):
    point_length = max_time
    time = 0
    count = 0
    data = np.zeros((max_time, 4))
    switch = 0
    measurement_points = np.linspace(0,time_length,point_length,endpoint=False)
    print (n_data)
    for j in range(max_bounce):
        # Is the velocity low enough to switch to rolling
        if np.abs(IC[3])<0.0001:
            switch = 1
        if switch:
            sol = solve_ivp(lambda t, X: f_roll(t, X, P), [time, time_length], IC, events=hit_ground, dense_output=True)
        else:
            sol = solve_ivp(lambda t, X: f(t, X, P), [time, time_length], IC, events=hit_ground, dense_output=True)
        impactState = sol.y[:,-1]
        time = sol.t[-1]
        points = measurement_points[np.logical_and(sol.t[0]<=measurement_points,measurement_points<sol.t[-1])]
        IC = calculate_IC(impactState, P[0])
        # If we don't have any points generated by the step move to the next iteration
        if (len(points)<1):
            continue
        X = sol.sol(points).T
        data_len = np.min((X.shape[0], max_time - count))
        data[count:count+data_len,:] = np.concatenate((X[:data_len,0].reshape(data_len,1), X[:data_len,1].reshape(data_len,1), X[:data_len,2].reshape(data_len,1), X[:data_len,3].reshape(data_len,1)),1)
        count+=data_len

    return data[::1]



def generate_trajs(parameters, n, max_bounce, max_time, time_length):
    data = np.zeros((n,max_time,4))
    i = 0
    par_count = 0


    inputs = range(n)
    num_cores = 1
    num_cores = multiprocessing.cpu_count()
    data = Parallel(n_jobs=num_cores)(delayed(generate_traj)(parameters[i, :4], parameters[i, 4:], max_bounce, max_time, i, time_length) for i in inputs)
    data = np.asarray(data)
    return data


def motion_blur_image(count, data, filt=False):
    import pdb; pdb.set_trace()
    images = np.zeros((data.shape[1]-1,data.shape[2],data.shape[3]))
    for i in range(images.shape[0]):
        img1 = data[count,i]
        img2 = data[count,i+1]
        diff = cv2.absdiff(img1, img2)
        images[i] = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        if (filt):
            kernel = np.ones((2,2),np.uint8)
            images[i] = cv2.morphologyEx(images[i], cv2.MORPH_OPEN, kernel)
            kernel = np.ones((4,4),np.uint8)
            images[i] = cv2.dilate(images[i], kernel,iterations = 10)
        thr = 5
        images[i][np.logical_and(images[i]>0, images[i]<thr)] = 0
        images[i][images[i]>=thr] = 1
    return images




def motion_blur_images(data, filt=False):
    inputs = range(data.shape[0])
    num_cores = 1
    num_cores = multiprocessing.cpu_count()
    images = Parallel(n_jobs=num_cores)(delayed(motion_blur_image)(i, data, filt) for i in inputs)
    images = np.asarray(images)

    return images



def generate_video(i, max_time, take_every_n_el, original, radius, data, reshape):
    print (i)
    count = 0
    images = np.zeros((max_time//take_every_n_el, reshape, reshape))
    # UNCOMMENT WHEN GENERATING VIDEOS
    for j in range(0, max_time, take_every_n_el):
        img = create_image(original=original, x=original-radius-data[i,j,0]*((original-2*radius)/10), y=original-radius-data[i,j,2]*((original-2*radius)/10), radius=radius, reshape=reshape)
        images[count, :, :] = img
        count += 1
    return images


def generate_images(data, n, max_time, take_every_n_el, bw, real=False, noise=False):
    reshape = 28
    original = 256
    radius = 30



    inputs = range(n)
    num_cores = 1
    num_cores = multiprocessing.cpu_count()
    if (bw):
        images = Parallel(n_jobs=num_cores)(delayed(generate_video)(i, max_time, take_every_n_el, original, radius, data, reshape) for i in inputs)
    else:
        #images = Parallel(n_jobs=num_cores)(delayed(generate_video_real)(data, i, max_time, take_every_n_el, noise=noise) for i in inputs)
        #images = generate_video_real(data, 0, max_time, take_every_n_el)
        pass

    images = np.asarray(images)

    """
    if (real==False):
        images = motion_blur_images(images) # 2x51x50x100x3
    """
    return images


def add_offset(trajs):
    offset = np.random.rand(trajs.shape[0])*5
    offset = np.expand_dims(offset, axis=1)
    offset = np.repeat(offset, trajs.shape[1], axis=1)
    trajs[:,:,2] += offset
    return trajs

def remove_frames(train_data):
    data_size = train_data.shape[0]
    for i in range(data_size):
        print (i)
        intervals = np.random.randint(1,5)
        lengths = np.random.randint(1,20, size=intervals)
        starts = np.random.randint(low=10,high=train_data.shape[1]-10,size=intervals)
        ends = starts+lengths
        for j in range(intervals):
            train_data[i,starts[j]:ends[j]]=0
    return train_data

def add_white_noise(train_data):
    noise = np.random.rand(*train_data.shape)
    noise_ratio = np.random.rand(*train_data.shape[:2])/5+0.8

    noise_ratio = np.expand_dims(noise_ratio, axis=3)
    noise_ratio = np.expand_dims(noise_ratio, axis=4)
    noise_ratio = np.repeat(noise_ratio, noise.shape[2], axis=2)
    noise_ratio = np.repeat(noise_ratio, noise.shape[3], axis=3)

    noise[np.where(noise<=noise_ratio)] = 0
    noise[np.where(noise>noise_ratio)] = 1
    return train_data+noise




def generate_data(n_train, n_test, name, full_phys, percentage_change=0):




    """
    Physics_Params = {
        "horizontal_position": [0, 10],
        "horizontal_velocity": [-8, -4, 4, 8], # 4 instead of 6
        "vertical_position": [6, 8],
        "vertical_velocity": [-4, -3, 3, 4], # 2 instead of 3
        "rest": [0.6, 1.0],
        "gravity": [6.81, 12.81],
        "drag_coefficient": [0.05, 0.0005], # 1 instead of 3
        "rolling_coefficient": [0.0, 0.7], # 5 instead of 7
    }


    phys_names = ["horizontal_position", "horizontal_velocity", "vertical_position", "vertical_velocity", "rest", "gravity", "drag_coefficient", "rolling_coefficient"]
    """


    Physics_Params = OrderedDict([('horizontal_position', [0, 10]), ('horizontal_velocity', [-8, -4, 4, 8]), ('vertical_position', [6, 8]), ('vertical_velocity', [-4, -3, 3, 4]), ('rest', [0.6, 1.0]), ('gravity', [6.81, 12.81]), ('drag_coefficient', [0.05, 0.0005]), ('rolling_coefficient', [0.0, 0.7])]).items()

    # MNIST
    interpolation_intervals = 10
    max_bounce = 300
    max_time = 200
    time_length = 10
    take_every_n_el = 1
    bw = True
    offset = False
    real = True
    noise = False
    white_noise = False
    remove_fr = False


    train_examples = n_train
    test_examples = n_test


    train_parameters = generate_parameters(Physics_Params, train_examples, interpolation_intervals, True)
    test_parameters = generate_parameters(Physics_Params, test_examples, interpolation_intervals, False)

    train_trajs = generate_trajs(train_parameters, train_examples, max_bounce, max_time, time_length)
    test_trajs = generate_trajs(test_parameters, test_examples, max_bounce, max_time, time_length)

    if (offset):
        train_trajs = add_offset(train_trajs)
        test_trajs = add_offset(test_trajs)

    train_data = generate_images(train_trajs, train_examples, max_time, take_every_n_el, bw=bw, real=real, noise=noise)
    test_data = generate_images(test_trajs, test_examples, max_time, take_every_n_el, bw=bw, real=real, noise=noise)

    video9 = np.zeros((200,3*28,3*28,3))
    for i in range(3):
        for j in range(3):
            video = np.expand_dims(train_data[i+3*j], axis=4)
            video[:,0,:,:] = 1
            video[:,-1,:,:] = 1
            video[:,:,0,:] = 1
            video[:,:,-1,:] = 1
            video = np.repeat(video, 3, axis=3)*255
            video9[:,i*28:(i+1)*28,j*28:(j+1)*28] = video
    skvideo.io.vwrite("outputvideo.mp4", video9.astype(np.uint8))


    np.savez('train_'+name, train_data, train_trajs, train_parameters, list(Physics_Params))
    np.savez('test_'+name, test_data, test_trajs, test_parameters, list(Physics_Params))


if __name__ == "__main__":
    generate_data(3000,100,'dataset',True)
