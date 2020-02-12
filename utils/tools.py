#from model import *
#from viz import *

from models.vrnn import *
from utils.viz import *
import seaborn as sns
import pandas as pd

def generate_data(T = 20, L = 2000, N = 200, sample=200):
    data = np.empty((N, L, 4), 'float64')
    params = np.empty(N, 'float64')
    for i in range(N):
        pos, vel, time, rest = ode_ball.bouncing_ball_traj(length=T, r=0.99, timesteps=L, ns=[0], ms=[L])
        data[i,:,0] = pos
        data[i,:,1] = vel*(L//sample)
        data[i,:,1] = vel
        data[i,:,2] = -9.8
        data[i,:,2] = rest
        params[i] = rest
    data = data[:,::L//sample,:]
    return data, params

def normalize_data(data):
    pos_min = data[:,:,0].min()
    pos_max = data[:,:,0].max()
    vel_min = data[:,:,1].min()
    vel_max = data[:,:,1].max()
    data[:,:,0] = (data[:,:,0]-pos_min)/(pos_max-pos_min)
    data[:,:,1] = (data[:,:,1]-vel_min)/(vel_max-vel_min)
    return data



def normalize_all_data(data, dyn, params, params_intervals):
    small_offset = 0.0000000
    for i, (name, interval) in enumerate(params_intervals):
        if (i==0):
            dyn[i] = (dyn[i]+small_offset)/10
        elif (i==1):
            dyn[i] = (dyn[i]+small_offset+8)/16
        elif (i==2):
            dyn[i] = (dyn[i]+small_offset)/10
        elif (i==3):
            dyn[i] = (dyn[i]+small_offset+15)/30
        if (len(params.shape)==2):
            params[:,i]=(params[:,i]+small_offset-interval[0])/(interval[1]-interval[0])
        else:
            params[:,:,i]=(params[:,:,i]+small_offset-interval[0])/(interval[1]-interval[0])
    # Don't include the velocity in the training currently
    #dyn = [dyn[0],dyn[2],dyn[1],dyn[3]]
    dyn = [dyn[0],dyn[2]]
    if (len(params.shape)==2):
        params = torch.cat((params[:,:1],params[:,2:3],params[:,4:]),1)
    else:
        params = torch.cat((params[:,:,:1],params[:,:,2:3],params[:,:,4:]),2)

    return dyn, params

def test_pred_model(data_loader, name, model, device, params_intervals, args, sw, epoch, decoder):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (input, output, x, vel_x, y, vel, time, params) in enumerate(data_loader):
            data = torch.cat((input,output),1)
            data = data.to(device).float()
            dyn = [x,vel_x,y,vel]
            dyn, params = normalize_all_data(data, dyn, params, params_intervals)
            n_predictions = 10
            interval = args.back_frames//n_predictions
            back_frames = args.back_frames
            forw_frames = args.forw_frames
            for j in range(n_predictions):
                args.back_frames = back_frames-interval*(n_predictions-j-1)
                args.forw_frames = forw_frames+interval*(n_predictions-j-1)
                kld_loss, nll_loss, phys_loss, enc, dec_pred, h, dec_mean_t, all_z_t_mean, all_z_t_var, all_z_t = model(data.view(data.shape[0],args.back_frames+args.forw_frames,-1)[:,:args.back_frames])
                # Future prob predictions
                visualize_latent_future_pred(model, data, args, h, dec_mean_t, dyn, params, sw, counter=epoch, name=name+"_"+str(j))
            test_loss += kld_loss + nll_loss + phys_loss
            # Prob predictions of the current interval
            visualize_latent_current_pred(model, data, args, h, dec_mean_t, dyn, params, sw, counter=epoch, name=name+"_current")
            kld_loss, nll_loss, phys_loss, enc, dec_filt, h, dec_mean_t, all_z_t, _, _ = model(data.view(data.shape[0],args.back_frames+args.forw_frames,-1)[:,args.back_frames:],h=h,pred=1)

            if (decoder):
                data_pred = data[:1,:args.back_frames]
                data_filt = data[:1,args.back_frames:]
                dec_pred_mean = dec_pred[0][0].view(1,-1,1,data.shape[-2],data.shape[-1])
                dec_filt_mean = dec_filt[0][0].view(1,-1,1,data.shape[-2],data.shape[-1])
                sw.add_video('dec_pred_'+name,torch.cat((data_pred,dec_pred_mean),0),epoch,fps=30)
                sw.add_video('dec_filt_'+name,torch.cat((data_filt,dec_filt_mean),0),epoch,fps=30)
            break

    test_loss /= len(data_loader.dataset)
    print('====> %s set loss: %f' % (name, test_loss))

def acc_latent_space(data_loader, name, model, device, params_intervals, args, sw, epoch, timelen):
    model.eval()
    test_loss = 0
    correct = np.zeros((6, data_loader.dataset.__len__(), args.back_frames+args.forw_frames))
    predicted = np.zeros((6, data_loader.dataset.__len__(), args.back_frames+args.forw_frames))
    count = 0
    with torch.no_grad():
        for i, (input, output, x, vel_x, y, vel, time, params) in enumerate(data_loader):
            data = torch.cat((input,output),1)
            data = data.to(device).float()
            dyn = np.stack((x.numpy(),vel_x.numpy(),y.numpy(),vel.numpy()))
            dyn, params = normalize_all_data(data, dyn, params, params_intervals)
            kld_loss, nll_loss, phys_loss, enc, dec_pred, h, dec_mean_t, all_z_t, _, _ = model(data.view(data.shape[0],args.back_frames+args.forw_frames,-1))
            params = np.asarray(params).T
            params = np.expand_dims(params, axis=2)
            params = np.tile(params, args.back_frames + args.forw_frames)
            correct[:2,count*args.batch_size:count*args.batch_size+data.shape[0]] = dyn
            correct[2:,count*args.batch_size:count*args.batch_size+data.shape[0]] = params[2:]
            predicted[:, count*args.batch_size:count*args.batch_size+data.shape[0]] = all_z_t.permute(2,0,1).cpu().numpy()[:6]
            count += 1

    difference = np.abs(correct-predicted)
    names = ["horizontal pos", "vertical pos", "restitution", "gravity", "drag", "roll"]
    figure = plt.figure(figsize=(10,2))
    for i in range(0,6):
        ax = plt.subplot(1,6,i+1)
        df = pd.DataFrame(difference[i]).melt()
        sns.lineplot(x="variable", y="value", data=df, ax=ax)
        ax.set_xlabel('timestep')
        ax.set_ylabel('error')
        ax.set_title(names[i])
        if (i!=5):
            ax.set_ylim(0,0.6)
        ax.set_xlim(0,timelen)
        #ax.set_aspect(aspect=1.0)
    plt.tight_layout()
    #plt.savefig('latent.pdf')
    sw.add_figure('latent_accuracy', figure, epoch)

def add_noise(percentage, params, params_intervals):
    for i, (name, interval) in enumerate(params_intervals):
        if i<3:
            continue
        difference = np.abs(interval[0]-interval[1])
        params[:,i-2]+=torch.from_numpy(np.random.uniform(-difference*percentage,difference*percentage,params.shape[0]))
    return params



