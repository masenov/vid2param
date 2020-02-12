import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def viz_data(pos,vel):
    plt.scatter(np.arange(len(pos)),np.asarray(pos), label="pos")
    plt.scatter(np.arange(len(vel)),np.asarray(vel), label="vel")
    plt.legend()
    plt.title("Restitution factor - 0.6")
    plt.show()

def viz_data_tb(pos_x,vel_x,pos,vel,rest,sw):
    for i in range(pos.size(0)):
        figure = plt.figure()
        plt.scatter(np.arange(len(pos[i,:])),np.asarray(pos[i,:].cpu()), label="pos")
        plt.scatter(np.arange(len(vel[i,:])),np.asarray(vel[i,:].cpu()), label="vel")
        plt.legend()
        plt.title("Params - "+str(rest[i].cpu().numpy()))
        sw.add_figure('example_data/y', figure, i)

        figure = plt.figure()
        plt.scatter(np.arange(len(pos_x[i,:])),np.asarray(pos_x[i,:].cpu()), label="pos")
        plt.scatter(np.arange(len(vel_x[i,:])),np.asarray(vel_x[i,:].cpu()), label="vel")
        plt.legend()
        plt.title("Params - "+str(rest[i].cpu().numpy()))
        sw.add_figure('example_data/x', figure, i)






def draw(yi, color, future):
    plt.plot(np.arange(len(yi)-future), yi[:len(yi)-future], color, linewidth = 2.0)
    plt.plot(np.arange(len(yi)-future, len(yi)), yi[-future:], color + ':', linewidth = 2.0)

def viz_results(pred, future, index):
    # draw the result
    plt.figure(figsize=(30,10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    colors = ['r','g','b','c','m','y']
    for i in range(6):
        draw(pred[i,:,0], colors[i], future)
    plt.savefig('small_predict%d.pdf'%index)
    plt.close()

def visualize_latent_pred(n_frames, dyn_gt, par_gt, all_pred, sw, count=1, counter=0, start=0, name="", images_all=None):
    if (name[-7:]=="current"):
        #figure = plt.figure(figsize=(15,3))
        pass
    else:
        import matplotlib
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        figure = plt.figure(figsize=(3,6))
    names = ["horizontal pos", "vertical pos", "restitution", "gravity", "drag", "roll"]

    if (name[-7:]=="current"):
        figure = plt.figure(figsize=(15,1.8), constrained_layout=True)
        widths = [2, 2, 2, 2, 1]
        heights = [1]
        gs = figure.add_gridspec(ncols=5, nrows=1, width_ratios=widths,
                          height_ratios=heights)
        #gs = gridspec.GridSpec(1, 5, width_ratios=[1, 2, 2, 2, 2])
    for i in range(count):
        pred = all_pred[i]
        for j in range(0,5):
            if (name[-7:]=="current"):
                ax = plt.subplot(gs[0,j], aspect="auto", adjustable='box')
            else:
                if (j>=2):
                    continue
                ax = plt.subplot(2,1,2-j, aspect="auto", adjustable='box')

            if (j<2 and name[-7:]!="current") or (j<4 and name[-7:]=="current"):
                if (name[-7:]!="current"):
                    plt.plot(start+np.arange(0,n_frames),pred[0,0:,j],'-.',c='r')
                    plt.xlim(0,200)
                    ax.set_xticks([200-n_frames,200])
                    ax.set_xticklabels([200-n_frames,200])
                    ax.set_yticklabels([])
                    """
                    plt.title(names[j])
                    plt.xlabel("timestep")
                    plt.ylabel("normalized value")
                    """
                else:
                    plt.plot(start+np.arange(0,n_frames),pred[0,0:,j+2],':',c='r')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                plt.ylim(0,1)



            if (i==count-1):
                if (j==4):
                    if (name[-7:]=="current"):
                        images = images_all[0,:,0].detach().cpu().numpy()
                        images = (images-np.min(images))/np.max(images)
                        final_image = np.zeros(images.shape[1:])
                        max_time = images.shape[0]
                        scale = np.ones(max_time)
                        scale = np.arange(max_time)
                        for k in range(max_time):
                            image = images[k]
                            final_image[image>0] = scale[k]
                        im = final_image
                        im /= np.sum(scale)
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        plt.axis('off')
                        plt.imshow(im)
                if (j<2):
                    if (name[-7:]!="current"):
                        plt.plot(start+np.arange(n_frames),dyn_gt[j][0,start:start+n_frames].cpu().numpy(),'--', c='g',linewidth=3.0)
                        plt.plot(np.arange(start+1),dyn_gt[j][0,:start+1].cpu().numpy(),c='g',linewidth=3.0)
                if (j<4 and name[-7:]=="current"):
                    if (len(par_gt.shape)==2):
                        plt.plot(np.ones(start+n_frames)*par_gt[0][j+2].item(),c='g',linewidth=3.0)
                    else:
                        length = (start+n_frames)//4
                        for k in range(4):
                            plt.plot(np.arange(length)+k*length,np.ones(length)*par_gt[0][k][j+2].item(),c='g',linewidth=3.0)



    for j in range(0,4):
        if (name[-7:]=="current"):
            all_pred_arr = np.zeros((count,*all_pred[0].shape))
            all_pred_arr = np.asarray(all_pred)
            ax = plt.subplot(gs[0,j], aspect="auto", adjustable='box')
            plt.plot(np.sum(all_pred_arr,0)[0,0:,j+2]/count,c='b',linewidth=3.0)
            if (len(par_gt.shape)==2):
                plt.plot(np.ones(start+n_frames)*par_gt[0][j+2].item(),c='g',linewidth=3.0)
            else:
                length = (start+n_frames)//4
                for k in range(4):
                    plt.plot(np.arange(length)+k*length,np.ones(length)*par_gt[0][k][j+2].item(),c='g',linewidth=3.0)



    plt.tight_layout()
    #plt.savefig(str(count)+'_'+name+'.pdf')
    plt.show()
    if (count==1):
        sw.add_figure('latent_space/identification_'+name, figure, counter)
    else:
        sw.add_figure('latent_space/prediction_'+name, figure, counter)
"""
def visualize_latent_pred(n_frames, dyn_gt, par_gt, all_pred, sw, count=1, counter=0, start=0, name=""):
    figure = plt.figure(figsize=(6,9))
    names = ["horizontal pos", "vertical pos", "restitution", "gravity", "drag", "roll"]
    pred = all_pred[0]
    var = all_pred[1]
    for j in range(0,6):
        plt.subplot(3,2,j+1)
        plt.plot(start+np.arange(0,n_frames),pred[0,0:,j],c='b')
        plt.fill_between(start+np.arange(0,n_frames),pred[0,0:,j]-var[0,0:,j],pred[0,0:,j]+var[0,0:,j],color='r')

        if (j<2):
            plt.plot(start+np.arange(n_frames),dyn_gt[j][0,start:start+n_frames].cpu().numpy(),c='y')
            plt.plot(np.arange(start+1),dyn_gt[j][0,:start+1].cpu().numpy(),c='g')
        else:
            #a = 0
            if (len(par_gt.shape)==2):
                plt.plot(np.ones(start+n_frames)*par_gt[0][j].item(),c='g')
            else:
                length = (start+n_frames)//4
                for k in range(4):
                    plt.plot(np.arange(length)+k*length,np.ones(length)*par_gt[0][k][j].item(),c='g')



        plt.ylim(0,1)
        plt.title(names[j])
        plt.xlabel("timestep")
        plt.ylabel("normalized value")

    plt.tight_layout()
    #plt.savefig(str(count)+'_'+name+'.pdf')
    plt.show()
    if (count==1):
        sw.add_figure('latent_space/identification_'+name, figure, counter)
    else:
        sw.add_figure('latent_space/prediction_'+name, figure, counter)

"""

def visualize_latent_future_pred(model, data, args, h, dec_mean_t, dyn, params, sw, counter=0, name=""):
    pred = []
    number_of_runs = 20
    for j in range(number_of_runs):
        kld_loss, nll_loss, phys_loss,  enc, dec_filt, _, _, all_z_t_mean, all_z_t_var, all_z_t = model(data.view(data.shape[0],args.back_frames+args.forw_frames,-1)[:,args.back_frames:],h=h,pred=dec_mean_t)
        shape = all_z_t_mean[0,:,1].cpu().numpy().shape[0]
        pred.append(all_z_t_mean.cpu().numpy())
    #pred = [all_z_t_mean.cpu().numpy(), all_z_t_var.cpu().numpy()]
    visualize_latent_pred(args.forw_frames, dyn, params, pred, sw, count=number_of_runs, counter=counter, start=args.back_frames, name=name, images_all=data)


def visualize_latent_current_pred(model, data, args, h, dec_mean_t, dyn, params, sw, counter=0, name=""):
    pred = []
    number_of_runs = 20
    for j in range(number_of_runs):
        kld_loss, nll_loss, phys_loss, enc, dec_pred, _, _, all_z_t_mean, all_z_t_var, all_z_t = model(data.view(data.shape[0],args.back_frames+args.forw_frames,-1)[:,:args.back_frames+args.forw_frames])
        pred.append(all_z_t.cpu().numpy())
    #pred = [all_z_t_mean.cpu().numpy(), all_z_t_var.cpu().numpy()]
    visualize_latent_pred(args.forw_frames+args.back_frames, dyn, params, pred, sw, count=number_of_runs, counter=counter, start=0, name=name, images_all=data)



