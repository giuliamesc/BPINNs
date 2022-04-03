## TO BE DEPRECATED
import numpy as np
from matplotlib import pyplot as plt
import os

"""
Define all the plot_results functions
"""

def plot_result(n_output_par, u_NN, f_NN, u_std, f_std, datasets_class, path_plot, add_name_plot=""):
    """
    plot mean and 2_std_bounds result
    """
    if (n_output_par == 1):

        # get domain dataset
        inputs,u_true,f_true = datasets_class.get_dom_data()


        if(datasets_class.get_n_input()==3): # 3D plot
            x = inputs[:,0]
            y = inputs[:,1]
            z = inputs[:,2]
            # datasets_class.n_1 = x.shape[0]
            # datasets_class.n_2 = 1
            # datasets_class.n_2 = 1
            xx = np.reshape( x, (datasets_class.n_1,datasets_class.n_2,datasets_class.n_3))
            yy = np.reshape( y, (datasets_class.n_1,datasets_class.n_2,datasets_class.n_3))
            zz = np.reshape( z, (datasets_class.n_1,datasets_class.n_2,datasets_class.n_3))

            # mean prediction
            uu_NN = np.reshape( u_NN, (datasets_class.n_1,datasets_class.n_2,datasets_class.n_3))
            ff_NN = np.reshape( f_NN, (datasets_class.n_1,datasets_class.n_2,datasets_class.n_3))
            # true output
            uu_true = np.reshape(u_true, (datasets_class.n_1,datasets_class.n_2,datasets_class.n_3))
            ff_true = np.reshape(f_true, (datasets_class.n_1,datasets_class.n_2,datasets_class.n_3))
            # std prediction
            uu_std = np.reshape(u_std, (datasets_class.n_1,datasets_class.n_2,datasets_class.n_3))
            ff_std = np.reshape(f_std, (datasets_class.n_1,datasets_class.n_2,datasets_class.n_3))

            k_1 = 5
            k_2 = 1
            k_3 = 1

            # Plot u (Real, Mean_NN and Standard Deviation)
            fig = plt.figure(figsize=(30,10))
            ax = fig.add_subplot(131, projection='3d')
            ax.set_title("Real noisy u")

            p = ax.scatter(xx[::k_1,::k_2,::k_3], yy[::k_1,::k_2,::k_3],
                           zz[::k_1,::k_2,::k_3],c=uu_true[::k_1,::k_2,::k_3])
            fig.colorbar(p)

            ax = fig.add_subplot(132, projection='3d')
            ax.set_title("NN prediction u")
            p = ax.scatter(xx[::k_1,::k_2,::k_3], yy[::k_1,::k_2,::k_3],
                           zz[::k_1,::k_2,::k_3],c=uu_NN[::k_1,::k_2,::k_3])
            fig.colorbar(p)

            ax = fig.add_subplot(133, projection='3d')
            ax.set_title("NN std u")
            p = ax.scatter(xx[::k_1,::k_2,::k_3], yy[::k_1,::k_2,::k_3],
                           zz[::k_1,::k_2,::k_3],c=uu_std[::k_1,::k_2,::k_3])
            fig.colorbar(p)
            name = add_name_plot + "u.png"
            path = os.path.join(path_plot,name)

            #######################################################################################################
            # Plot velocity (Real, Mean_NN and Standard Deviation)
            fig.savefig(path)
            fig = plt.figure(figsize=(30,10))
            ax = fig.add_subplot(131, projection='3d')
            ax.set_title("Real noisy velocities")
            p = ax.scatter(xx[::k_1,::k_2,::k_3], yy[::k_1,::k_2,::k_3],
                           zz[::k_1,::k_2,::k_3],c=ff_true[::k_1,::k_2,::k_3])
            fig.colorbar(p)

            ax = fig.add_subplot(132, projection='3d')
            ax.set_title("NN prediction velocities")
            p = ax.scatter(xx[::k_1,::k_2,::k_3], yy[::k_1,::k_2,::k_3],
                           zz[::k_1,::k_2,::k_3],c=ff_NN[::k_1,::k_2,::k_3])
            fig.colorbar(p)

            ax = fig.add_subplot(133, projection='3d')
            ax.set_title("NN std velocities")
            p = ax.scatter(xx[::k_1,::k_2,::k_3], yy[::k_1,::k_2,::k_3],
                           zz[::k_1,::k_2,::k_3],c=ff_std[::k_1,::k_2,::k_3])
            fig.colorbar(p)
            name = add_name_plot + "f.png"
            path = os.path.join(path_plot,name)
            fig.savefig(path)


        elif(datasets_class.get_n_input()==2): # 2D plot
            x = inputs[:,0]
            y = inputs[:,1]

            # Plot arguments for title
            plot_x = 0.4*np.max(x)
            plot_y = 0.95*np.max(y)
            fontsize = 18

            # Load the sparse data used for training -> 100 true observations
            inputs_train,_,_ = datasets_class.get_exact_data()
            xtrain = inputs_train[:,0]
            ytrain = inputs_train[:,1]

            # Plot u (Real, Mean_NN and Standard Deviation)

            # plot real+noise vs NN mean vs NN std
            fig = plt.figure()
            fig.set_size_inches((15,5))

            plt.subplot(1,3,1)
            plt.scatter(x, y, c= u_true, label = 'u true', cmap = 'coolwarm', vmin = min(u_true), vmax = max(u_true))
            plt.colorbar()
            plt.scatter(xtrain, ytrain, marker = 'x', c = 'black')
            plt.text(plot_x, plot_y, r'at true', {'color': 'b', 'fontsize': fontsize})
            plt.axis('equal')
            plt.xlim([0,1])
            plt.ylim([0,1])

            n_1 = datasets_class.n_1
            n_2 = datasets_class.n_2

            xx = x.reshape((n_1,n_2))
            yy = y.reshape((n_1,n_2))
            uu = u_true.reshape((n_1,n_2))
            plt.contour(xx, yy, uu, levels=20, colors="black", alpha=0.5)

            plt.title(' u REAL + NOISE')

            plt.subplot(1,3,2)
            plt.scatter(x, y, c= u_NN, label = 'u_NN', cmap = 'coolwarm', vmin = min(u_NN), vmax = max(u_NN))
            plt.text(plot_x, plot_y, r'at Mean', {'color': 'b', 'fontsize': fontsize})
            plt.colorbar()
            plt.axis('equal')
            plt.xlim([0,1])
            plt.ylim([0,1])
            tt = u_NN.reshape((n_1,n_2))
            plt.contour(xx, yy, tt, levels=20, colors="black", alpha=0.5)
            plt.title('u NN MEAN')

            plt.subplot(1,3,3)
            plt.scatter(x, y, c=u_std , label = 'u_NN_std', cmap = 'coolwarm', vmin = 0.0, vmax = max(u_std))
            plt.text(plot_x, plot_y, r'at Std', {'color': 'b', 'fontsize': fontsize})
            plt.axis('equal')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.colorbar()
            plt.title('u NN STD')

            plt.tight_layout()
            name = add_name_plot + "u.png"
            path = os.path.join(path_plot,name)
            plt.savefig(path,bbox_inches = 'tight')


            # Plot velocity (Real, Mean_NN and Standard Deviation)
            fig = plt.figure()
            fig.set_size_inches((15,5))

            plt.subplot(1,3,1)
            plt.scatter(x, y, c= f_true, label = 'f true', cmap = 'coolwarm', vmin = min(f_true), vmax = max(f_true))
            plt.colorbar()
            plt.text(plot_x, plot_y, r'v true', {'color': 'b', 'fontsize': fontsize})
            plt.axis('equal')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.title('f REAL + NOISE')

            plt.subplot(1,3,2)
            plt.scatter(x, y, c= f_NN, label = 'f_NN', cmap = 'coolwarm', vmin = min(f_NN), vmax = max(f_NN))
            plt.text(plot_x, plot_y, r'v Mean', {'color': 'b', 'fontsize': fontsize})
            plt.colorbar()
            plt.axis('equal')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.title('f NN MEAN')

            plt.subplot(1,3,3)
            plt.scatter(x, y, c= f_std, label = 'ftd_NN_std', cmap = 'coolwarm', vmin = 0.0, vmax = max(f_std))
            plt.text(plot_x, plot_y, r'v Std', {'color': 'b', 'fontsize': fontsize})
            plt.axis('equal')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.colorbar()
            plt.title('f NN STD')

            plt.tight_layout()
            name = add_name_plot + "f.png"
            path = os.path.join(path_plot,name)
            plt.savefig(path,bbox_inches = 'tight')

        else: #1D
            x = inputs[:,0]

            # Load the sparse data used for training -> 100 true observations
            inputs_train,at_train,_ = datasets_class.get_exact_data_with_noise()
            xtrain = inputs_train[:,0]

            # Plot u (Real, Mean_NN and Standard Deviation buonds)
            plt.figure()
            plt.plot(x, u_NN, 'b--', label='mean')
            plt.plot(x, u_NN-u_std, 'g--', label='mean-std')
            plt.plot(x, u_NN+u_std, 'g--', label='mean+std')
            plt.plot(x, u_true, 'r-', label='true')
            plt.plot(xtrain, at_train, 'r*')

            plt.xlabel('x')
            plt.ylabel('u')
            plt.legend(prop={'size': 9})
            plt.title('Confidence Interval for u')
            name = add_name_plot + "u_axis.png"
            path = os.path.join(path_plot,name)
            plt.savefig(path,bbox_inches= 'tight')

            # Plot velocity (Real, Mean_NN and Standard Deviation bounds)
            plt.figure()
            plt.plot(x, f_NN, 'b--', label='mean')
            plt.plot(x, f_NN-f_std, 'g--', label='mean-std')
            plt.plot(x, f_NN+f_std, 'g--', label='mean+std')
            plt.plot(x, f_true, 'r-', label='true')

            plt.xlabel('x')
            plt.ylabel('f')
            plt.legend(prop={'size': 9})
            plt.title('Confidence Interval for f')
            name = add_name_plot + "f_axis.png"
            path = os.path.join(path_plot,name)
            plt.savefig(path,bbox_inches= 'tight')



def plot_all_result(x, u, f, u_NN, f_NN, datasets_class, n_input, n_output_vel, method, path_plot, add_name_plot=""):
    """
    Plot all the samples:
    - If method == HMC : plot all the M samples
    - If method == SVGD : plot all the num_neural_networks prediction
    """
    if(n_input == 1):  # 1D plot
        inputs_train,u_train,_ = datasets_class.get_exact_data_with_noise()
        xtrain = inputs_train[:,0]

        plt.figure()

        if(method == "SVGD"):
            for i in range(u_NN.shape[1]):
                plt.plot(x, u_NN[:,i])
        elif(method == "HMC"):
            for i in range(u_NN.shape[0]):
                plt.plot(x, u_NN[i,:,0], 'b-',markersize=0.01, alpha=0.01)
        else:
            print("No method")

        plt.plot(x, u[:,0], 'r--', label='true')
        plt.plot(xtrain, u_train[:,0], 'r*')

        #plt.xlim([0,1])
        #plt.ylim([-0.5,1.5])
        plt.xlabel('x')
        plt.ylabel('Solution u')
        plt.legend(prop={'size': 9})
        if(method == 'SVGD'):
            plt.title('Output u of all the particles')
        elif(method == 'HMC'):
            plt.title('Samples from u reconstructed distribution')
        name = add_name_plot + "at_all.png"
        path = os.path.join(path_plot,name)
        plt.savefig(path,bbox_inches= 'tight')

        plt.figure()
        if(method == "SVGD"):
            for i in range(f_NN.shape[1]):
                plt.plot(x, f_NN[:,i])
        elif(method == "HMC"):
            for i in range(f_NN.shape[0]):
                plt.plot(x, f_NN[i,:], 'b-',markersize=0.01, alpha=0.01)
        else:
            print("No method")

        plt.plot(x, f[:,0], 'r--', label='true')

        plt.xlabel('x')
        plt.ylabel('Parametric field f')
        plt.legend(prop={'size': 9})
        if(method == 'SVGD'):
            plt.title('Output f of all the particles')
        elif(method == 'HMC'):
            plt.title('Samples from f reconstructed distribution')
        name = add_name_plot + "cv_all.png"
        path = os.path.join(path_plot,name)
        plt.savefig(path,bbox_inches= 'tight')

    else: # 2D plot
        plt.figure()
        if(method == "SVGD"):
            for i in range(u_NN.shape[1]):
                plt.plot(x, u_NN[:,i])
        elif(method == "HMC"):
            for i in range(u_NN.shape[0]):
                plt.plot(x, u_NN[i,:,0], 'b-',markersize=0.01, alpha=0.01)
        else:
            print("No method")

        plt.plot(x, u[:,0], 'r--', label='true')

        plt.xlabel('x')
        plt.ylabel('Solution u')
        plt.legend(prop={'size': 9})
        name = add_name_plot + "u_all.png"
        path = os.path.join(path_plot,name)
        plt.savefig(path,bbox_inches= 'tight')

        name_f = {0:"a",1:"b",2:"c"}

        if(method == "SVGD"):
            for k in range(f_NN.shape[2]):
                plt.figure()
                for i in range(f_NN.shape[1]):
                    plt.plot(x, f_NN[:,i,k])

                plt.plot(x, f[:,k], 'r--', label='true')

                plt.xlabel('x=y')
                name = 'f '
                if(f_NN.shape[2]>1):
                    name+= f[k]
                plt.ylabel(name)
                plt.legend(prop={'size': 9})
                name = add_name_plot + "f_all_"+name_f[k]+".png"
                path = os.path.join(path_plot,name)
                plt.savefig(path,bbox_inches= 'tight')

        elif(method == "HMC"):
            for k in range(f_NN.shape[3]):
                plt.figure()
                for i in range(f_NN.shape[0]):
                    plt.plot(x, f_NN[i,:,0,k],'b-',markersize=0.01, alpha=0.01)

                plt.plot(x, f[:,k], 'r--', label='true')

                plt.xlabel('x=y')
                name = 'f '
                if(f_NN.shape[2]>1):
                    name+= name_f[k]
                plt.ylabel(name)
                plt.legend(prop={'size': 9})
                name = add_name_plot + "f_all_"+name_f[k]+".png"
                path = os.path.join(path_plot,name)
                plt.savefig(path,bbox_inches= 'tight')

        else:
            print("No method")




def plot_axis_example(n_output_vel, datasets_class, bayes_nn, path_plot, add_name_plot=""):
    """
    Plot axis example: plot the solution along the axis y=x
    """
    ### plot the solution along the line y=x
    if(n_output_vel == 1):
        inputs,u,f = datasets_class.get_axis_data()
        x = inputs[:,0]
        u_NN, f_NN, u_std, f_std = bayes_nn.mean_and_std(inputs)

        u_NN = u_NN[:,0]
        u_std = u_std[:,0]
        f_NN = f_NN[:,0]
        f_std = f_std[:,0]


        plt.figure()
        plt.plot(x, u_NN, 'b--', label='mean')
        plt.plot(x, u_NN-u_std, 'g--', label='mean-std')
        plt.plot(x, u_NN+u_std, 'g--', label='mean+std')
        plt.plot(x, u, 'r-', label='true')

        plt.xlabel('x=y')
        plt.ylabel('u')
        plt.legend(prop={'size': 9})
        name = add_name_plot + "u_axis.png"
        path = os.path.join(path_plot,name)
        plt.savefig(path,bbox_inches= 'tight')

        plt.figure()
        plt.plot(x, f_NN, 'b--', label='mean')
        plt.plot(x, f_NN-f_std, 'g--', label='mean-std')
        plt.plot(x, f_NN+f_std, 'g--', label='mean+std')
        plt.plot(x, f, 'r-', label='true')

        plt.xlabel('x=y')
        plt.ylabel('f')
        plt.legend(prop={'size': 9})
        name = add_name_plot + "f_axis.png"
        path = os.path.join(path_plot,name)
        plt.savefig(path,bbox_inches= 'tight')


def plot_losses(LOSSD, LOSS1, LOSS, path_plot):
    """
    Plot log(losses)
    """
    plt.figure()
    plt.plot(np.log(LOSS),  "k--", lw=2.0, alpha=1.0, label = 'Loss Total')
    plt.plot(np.log(LOSSD), "r-" , lw=1.0, alpha=0.7, label = 'Loss Fitting')
    plt.plot(np.log(LOSS1), "b-" , lw=1.0, alpha=0.7, label = 'Loss Collocation')
    plt.xlabel('epochs')
    plt.ylabel('LogLoss')
    plt.legend(prop={'size': 9})
    path = os.path.join(path_plot,"loss.png")
    plt.savefig(path, bbox_inches= 'tight')

def plot_log_betas(log_betaD, log_betaR, path_plot):
    """
    Plot all the log_betas
    """
    plt.figure()
    for i in range(log_betaD.shape[1]):
        plt.plot(log_betaD[:,i])
    plt.xlabel('epochs')
    plt.ylabel('log_betaD')
    path = os.path.join(path_plot,"log_betaD.png")
    plt.savefig(path,bbox_inches= 'tight')

    plt.figure()
    for i in range(log_betaR.shape[1]):
        plt.plot(log_betaR[:,i])
    plt.xlabel('epochs')
    plt.ylabel('log_betaR')
    path = os.path.join(path_plot,"log_betaR.png")
    plt.savefig(path,bbox_inches= 'tight')

def plot_log_prob(eikonal_logloss, data_logloss, prior_logloss, path_plot):
    """
    Plot all the three posteriors components
    """
    plt.figure()
    for i in range(eikonal_logloss.shape[1]):
        plt.plot(eikonal_logloss[:,i])
    plt.savefig(os.path.join(path_plot, "eikonal_logloss.png"))

    plt.figure()
    for i in range(data_logloss.shape[1]):
        plt.plot(data_logloss[:,i])
    plt.savefig(os.path.join(path_plot, "data_logloss.png"))

    plt.figure()
    for i in range(prior_logloss.shape[1]):
        plt.plot(prior_logloss[:,i])
    plt.savefig(os.path.join(path_plot, "prior_logloss.png"))
