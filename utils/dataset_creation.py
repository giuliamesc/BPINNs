# %% Import Standard Packages

import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append("data") #Add the path to the datasets

## CONTROLLA DOVE ELIMINARE
def check_square(n):    #check if n is a perfect square
    return n==(math.floor(math.sqrt(n))**2)

# %% Start Main Class

class dataset_class:
    """
    Class for building the datasets (Domain, collocation and exact(with noise) ):
    Input:
        - par, a param object that store all the parameters

    Objects:
        - pde_type = "Isotropic" or "Anisotropic"
        - name_example = name of dataset we want to build/load
        - prop_exact = proportion of domain data to use as exact data
        - prop_collocation = proportion of domain data to use as collocation data
        - n_input = 1,2 or 3 (1D, 2D or 3D experiment)
        - noise_lv = noise level in exact data

    Private:
        - _flag_dataset_build = flag to build/load the dataset just one time
        - _flag_dataset_noise = flag to add the noise to exact data just one time

        (only if analytical)
            - n_domain = number of domain points
    """

    def __init__(self,par):
        """The constructor"""
        self.pde_type = par.pde
        self.name_example = par.experiment["dataset"]    # name of dataset to use

        self.prop_exact = par.experiment["prop_exact"] # prop of exact data
        self.prop_collocation = par.experiment["prop_collocation"] # prop of exact data
        self.n_input = par.n_input # 1D,2D,3D

        ## MODIFICARE
        self.n_output_vel = par.n_output_vel
        self.noise_lv = par.experiment["noise_lv"]

        self._flag_dataset_build = False
        self._flag_dataset_noise = False
        np.random.seed(par.param["random_seed"])

        self.n_domain = 0
        self.n_collocation = 0
        self.n_exact = 0

# %% Main Functions

    def _load_dataset(self):
        """load data from dataset"""
        path = os.path.join("data",self.name_example)

        if(self.n_input in [1,2,3]):
            x = np.load(os.path.join(path,"x.npy"))[...,None]
            inputs = x
            if(self.n_input > 1):
                y = np.load(os.path.join(path,"y.npy"))[...,None]
                inputs = np.concatenate((inputs,y),axis=1)
            if(self.n_input == 3):
                z = np.load(os.path.join(path,"z.npy"))[...,None]
                inputs = np.concatenate((inputs,z),axis=1)
            inputs = inputs.astype(np.float32)

            u = np.load(os.path.join(path,"u.npy")).astype(np.float32)
            f = np.load(os.path.join(path,"f.npy")).astype(np.float32)

            u = u[...,None]   # from shape (n_coll, ) -> (n_coll, 1)
            if(len(f.shape)==1):
                f = f[...,None] # add the last dimension only if we are in Isotropic case

            # store domain datasets
            self.inputs_dom = inputs
            self.U_dom = u
            self.F_dom = f

            #n_domain is simply the length of one of the dataset, x for instance
            self.n_domain = len(x)
            index = list(range(self.n_domain))


            # compute n_collocation using n_domain and prop_collocation (min 10)
            self.n_collocation = max(int(self.n_domain*self.prop_collocation),10)
            # build collocation dataset from domain dataset
            np.random.shuffle(index)
            index_coll = index[:self.n_collocation]
            self.inputs_coll = inputs[index_coll,:]
            self.U_coll = u[index_coll,:]
            self.F_coll = f[index_coll,:]

            # compute n_exact using n_domain and prop_exact (min 3)
            self.n_exact = max(int(self.n_domain*self.prop_exact),3)
            # build exact dataset from domain dataset
            np.random.shuffle(index)
            index_exac = index[:self.n_exact]
            self.inputs_exact = inputs[index_exac,:]
            self.U_exact = u[index_exac,:]
            self.F_exact = f[index_exac,:]

        else:
            raise Exception("Only case 1D, 2D, 3D allowed")


    def build_dataset(self):
        """
        Build dataset:
        call the functions to build the dataset
        """
        if(self._flag_dataset_build == False):  # we build the dataset only the first time
            self._load_dataset()
            self._flag_dataset_build = True

    def build_noisy_dataset(self):
        """
        Add noise to exact data
        """
        self.build_dataset()
        if(self._flag_dataset_noise == False):  # we add the noise only the first time
            self.U_with_noise = np.zeros_like(self.U_exact)
            self.F_with_noise = np.zeros_like(self.F_exact)

            for i in range(0,len(self.U_exact)):
                at_error = np.random.normal(0, self.noise_lv, 1)
                self.U_with_noise[i,:] = self.U_exact[i,:] + at_error

                # not so useful... we use onlt U_with_noise, not F (left it for future different implmentations)
                v_error = np.random.normal(0, self.noise_lv, self.F_exact.shape[1]) #1, 2 or 3
                self.F_with_noise[i,:] = self.F_exact[i,:] + v_error

            self._flag_dataset_noise = True

# %% Getters

    def get_dom_data(self):
        """ return domain data"""
        self.build_dataset()
        return self.inputs_dom, self.U_dom, self.F_dom

    def get_coll_data(self):
        """Return collocation data"""
        self.build_dataset()
        return self.inputs_coll,self.U_coll,self.F_coll

    def get_exact_data(self):
        """ return exact data"""
        self.build_dataset()
        return self.inputs_exact, self.U_exact, self.F_exact

    def get_exact_data_with_noise(self):
        """ return exact data + noise """
        self.build_dataset()
        self.build_noisy_dataset()
        return self.inputs_exact, self.U_with_noise, self.F_with_noise

    ## CONTROLLARE SE SOLO PER ANALITICO
    def save_dataset(self, save_path):
        """Save the dataset already built in save_path"""
        self.build_dataset()

        inputs,u,f = self.get_dom_data()
        final_path_name = os.path.join(save_path, "domain.npz")
        np.savez(final_path_name, inputs=inputs, u=u, f=f)

        inputs,u,f = self.get_coll_data()
        final_path_name = os.path.join(save_path, "collocation.npz")
        np.savez(final_path_name, inputs=inputs, u=u, f=f)

        inputs,u,f = self.get_exact_data()
        final_path_name = os.path.join(save_path, "exact.npz")
        np.savez(final_path_name, inputs=inputs, u=u, f=f)


    def get_num_collocation(self):
        """get number of collocation points"""
        self.build_dataset()
        return self.n_collocation

    def get_num_exact(self):
        """get number of exact points"""
        self.build_dataset()
        return self.n_exact

    def get_n_input(self):
        """get n input"""
        return self.n_input
        
# %% Plotters

    def plot(self, save_path = "plot"):
        """Plot the exact experiment"""
        self.build_dataset()
        self.build_noisy_dataset()
        if(self.n_input == 1):
            self._plot_1D(save_path) # 1D plot
        elif(self.n_input == 2):
            self._plot_2D(save_path) # 2D plot
        elif(self.n_input == 3):
            self._plot_3D(save_path) # 3D plot
        else:
            raise Exception("Only case 1D, 2D, 3D allowed")


    def _plot_1D(self, save_path = "plot"):
        """plot the 1D example"""
        x, u, f = self.get_dom_data()
        x_e, u_e, f_e = self.get_exact_data()
        fig, axs = plt.subplots(1,2,figsize = (12,5))

        axs[0].plot(x, u)
        axs[0].plot(x_e, u_e, 'r*')
        axs[0].set_title("u domian data with marked values")

        axs[1].plot(x, f)
        axs[1].set_title("f domain data")

        final_path_name = os.path.join(save_path, "exact.png")
        plt.savefig(final_path_name)


    def _plot_2D(self, save_path = "plot"):
        """plot the 2D example"""
        inputs,u,f = self.get_dom_data()
        inputs_e,u_e,f_e = self.get_exact_data()

        x = inputs[:,0]
        y = inputs[:,1]
        x_e = inputs_e[:,0]
        y_e = inputs_e[:,1]

        fig = plt.figure()
        fig.set_size_inches((12,5))
        plt.subplot(1,2,1)
        plt.scatter(x, y, c=u, label = 'u', cmap = 'coolwarm', vmin = min(u), vmax = max(u))
        plt.colorbar()

        xx = x.reshape((self.n_1,self.n_2))
        yy = y.reshape((self.n_1,self.n_2))
        tt = u.reshape((self.n_1,self.n_2))

        plt.contour(xx, yy, tt, levels=20, colors="black", alpha=0.5)
        plt.scatter(x_e, y_e, facecolors = 'none', edgecolor = 'k')
        plt.axis("equal")
        plt.xlim([0,np.max(x)])
        plt.ylim([0,np.max(y)])

        plt.subplot(1,2,2)
        plt.scatter(x, y, c=f, label = 'f', cmap = 'coolwarm', vmin = min(f), vmax = max(f))
        plt.colorbar()
        plt.axis("equal")
        plt.xlim([0,np.max(x)])
        plt.ylim([0,np.max(y)])

        final_path_name = os.path.join(save_path, "exact.png")
        plt.savefig(final_path_name)


    def _plot_3D(self, save_path):
        """plot the 3D example"""
        inputs,u,f = self.get_dom_data()

        x = inputs[:,0]
        y = inputs[:,1]
        z = inputs[:,2]

        xx = np.reshape( x, (self.n_1,self.n_2,self.n_3))
        yy = np.reshape( y, (self.n_1,self.n_2,self.n_3))
        zz = np.reshape( z, (self.n_1,self.n_2,self.n_3))
        tt = np.reshape( u, (self.n_1,self.n_2,self.n_3))
        vv = np.reshape( f, (self.n_1,self.n_2,self.n_3))

        fig = plt.figure(figsize=(15,20))
        ax = fig.add_subplot(211, projection='3d')
        k_1 = 5
        k_2 = 1
        k_3 = 1
        p = ax.scatter(xx[::k_1,::k_2,::k_3], yy[::k_1,::k_2,::k_3],
                        zz[::k_1,::k_2,::k_3],c=tt[::k_1,::k_2,::k_3])
        fig.colorbar(p)

        ax = fig.add_subplot(212, projection='3d')
        p = ax.scatter(xx[::k_1,::k_2,::k_3], yy[::k_1,::k_2,::k_3],
                        zz[::k_1,::k_2,::k_3],c=vv[::k_1,::k_2,::k_3])
        fig.colorbar(p)

        final_path_name = os.path.join(save_path, "exact.png")
        plt.savefig(final_path_name)

    def plot_domain(self, save_path = "plot/"):
        """plot only the domain"""
        if(self.n_input == 2):
            inputs, _, _ = self.datasets_class.get_dom_data()
            x = inputs[:,0]
            y = inputs[:,1]
            plt.figure()
            plt.scatter(x,y)
            plt.axis('equal')
            path = os.path.join(save_path,"domain.png")
            plt.savefig(path,bbox_inches = 'tight')
        elif(self.n_input == 3):
            inputs,_,_ = self.datasets_class.get_dom_data()
            x = inputs[:,0]
            y = inputs[:,1]
            z = inputs[:,2]
            xx = np.reshape( x, (self.n_1,self.n_2,self.n_3))
            yy = np.reshape( y, (self.n_1,self.n_2,self.n_3))
            zz = np.reshape( z, (self.n_1,self.n_2,self.n_3))

            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111, projection='3d')
            k = 2
            p = ax.scatter(xx[::k,::k,::k], yy[::k,::k,::k], zz[::k,::k,::k])
            fig.colorbar(p)
            path = os.path.join(save_path,"domain.png")
            fig.savefig(path,bbox_inches = 'tight')
