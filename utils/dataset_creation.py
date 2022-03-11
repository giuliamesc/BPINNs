
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Add the path to the datasets
"""
sys.path.append("data")

def check_square(n):    #check if n is a perfect square
    return n==(math.floor(math.sqrt(n))**2)


class dataset_class:
    """
    Class for building the datasets (Domain, collocation and exact(with noise) ):
    Input:
        - par, a param object that store all the parameters

    Objects:
        - pde_type = "Isotropic" or "Anisotropic"
        - name_example = name of dataset we want to build/load
        - dataset_type = analytical (where we know the exact solution) or dataset (from Pykonal experiment)
        - prop_exact = proportion of domain data to use as exact data
        - prop_collocation = proportion of domain data to use as collocation data
        - n_input = 1,2 or 3 (1D, 2D or 3D experiment)
        - n_output_vel = 1 if isotropic, >1 if anisotropic, dimension of conduction_velocity
        - noise_lv = noise level in exact data

    Private:
        - _flag_dataset_build = flag to build/load the dataset just one time
        - _flag_dataset_noise = flag to add the noise to exact data just one time

        (only if analytical)
            - n_domain = number of domain points
            - bool_collocation_uniform = True if we want a uniform grid for collocation
            - bool_exact_uniform = True if we want a uniform grid for exact

        (only if anisotropic)
        - an_constraints = store the constraints between the entries of the anisotropic conduction_velocity tensor
    """

    def __init__(self,par):
        """The constructor"""
        self.pde_type = par.pde  # Isotropic or Anisotropic
        self.name_example = par.experiment["dataset"]    # name of dataset to use
        self.dataset_type = par.dataset_type    # analytical functions or pykonal dataset

        self.prop_exact = par.experiment["prop_exact"] # prop of exact data
        self.prop_collocation = par.experiment["prop_collocation"] # prop of exact data
        self.n_input = par.n_input # 1D,2D,3D
        self.n_output_vel = par.n_output_vel
        self.noise_lv = par.experiment["noise_lv"]
        self._flag_dataset_build = False
        self._flag_dataset_noise = False
        np.random.seed(par.param["random_seed"])

        self.n_domain = 0
        self.n_collocation = 0
        self.n_exact = 0

        if(self.dataset_type == "analytical"):
            self.n_domain = par.experiment["n_domain"]
            self.n_collocation = int(self.n_domain*self.prop_collocation)
            self.n_exact = int(self.n_domain*self.prop_exact)
            self.bool_collocation_uniform = False#bool_collocation_uniform    # flag for collocation Uniform (True) or Random (False)
            self.bool_exact_uniform = par.experiment["is_uniform_exact"]    # flag for exact Uniform (True) or Random (False)

        if(self.pde_type == "anisotropic"):
            self.an_constraints = np.ones((self.n_output_vel-1))




    def _uniform_inputs(self, n_per_edge, n_other):
        """Build uniform inputs"""
        if(self.n_input==1):
            x = np.linspace(0,1,n_per_edge) # uniform points
            inputs = x[...,None]

        elif(self.n_input==2):
            x = np.linspace(0,1,n_per_edge) # n_per_edge is the floor(sqrt(n))
            y = np.linspace(0,1,n_per_edge)
            X_m, Y_m = np.meshgrid(x,y) # build the uniform mesh
            x = X_m.flatten()
            y = Y_m.flatten()
            if(n_other>0):  # if n was not a perfect square we have some points left
                x_more = np.random.random(size = (1,n_other)) # the points left will be assigned randomly
                y_more = np.random.random(size = (1,n_other))
                x = np.append(x, x_more)
                y = np.append(y, y_more)
            x = x[...,None]
            y = y[...,None]
            inputs = np.concatenate((x,y),axis=1)
        elif(self.n_input == 3):
            x = np.linspace(0,1,n_per_edge)
            y = np.linspace(0,1,n_per_edge)
            z = np.linspace(0,1,n_per_edge)
            X_m, Y_m, Z_m = np.meshgrid(x,y,z)
            x = X_m.flatten()
            y = Y_m.flatten()
            z = Z_m.flatten()
            if(n_other>0):
                x_more = np.random.random(size = (n_other,1))
                y_more = np.random.random(size = (n_other,1))
                z_more = np.random.random(size = (n_other,1))
                x = np.append(x, x_more)
                y = np.append(y, y_more)
                z = np.append(z, z_more)
            x = x[...,None]
            y = y[...,None]
            z = z[...,None]
            inputs = np.concatenate((x,y,z),axis=1)

        return inputs


    def _build_inputs(self,flag_uniform, n):
        """Build inputs (1D,2D or 3D)"""
        if flag_uniform == True:   #uniform grid

            if(self.n_input == 1):
                n_per_edge = n
                n_other = 0
            elif(self.n_input == 2):
                n_per_edge = math.floor(math.sqrt(n))
                n_other = n - (n_per_edge**2)   # remainings points
            elif(self.n_input == 3):
                n_per_edge = math.floor(n**(1./3))
                n_other = n - (n_per_edge**3)
            else:
                raise Exception("Only case 1D, 2D, 3D allowed")

            inputs = self._uniform_inputs(n_per_edge, n_other)

        else: #random grid
            if(self.n_input in [1,2,3]):   # build a grid of random points
                inputs = np.random.random(size = (n, self.n_input))
            else:
                raise Exception("Only case 1D, 2D, 3D allowed")

        inputs = inputs.astype(np.float32)
        return inputs



    def _build_coll_data(self):
        """build the collocation dataset"""
        #collocation
        inputs = self._build_inputs(self.bool_collocation_uniform, self.n_collocation)
        T_coll = self.AT(inputs)
        CV_coll = self.CV(inputs)

        # store collocation datasets
        self.inputs_coll = inputs
        self.T_coll = T_coll
        self.CV_coll = CV_coll


    def _build_exact_data(self):
        """build the exact dataset"""
        #exact
        # x = np.linspace(0.1,0.3,int(self.n_exact/2)) # uniform points
        # xx = np.linspace(0.7,0.9,int(self.n_exact/2)) # uniform points
        # x = np.concatenate((x,xx))
        # inputs = x[...,None]
        # inputs = inputs.astype(np.float32)
        inputs = self._build_inputs(self.bool_exact_uniform, self.n_exact)
        # inputs_1 = self._build_inputs(self.bool_exact_uniform, self.n_exact)
        # inputs_2 = np.linspace(0.0,1.0,2)
        # inputs_2 = inputs_2[...,None]
        # inputs_2 = inputs_2.astype(np.float32)
        # inputs = np.concatenate((inputs_1, inputs_2))

        T_exact = self.AT(inputs)
        CV_exact = self.CV(inputs)

        # store exact datasets
        self.inputs_exact = inputs
        self.T_exact = T_exact
        self.CV_exact = CV_exact


    def _build_dom_data(self):
        """build the domain dataset (used only for plot and final results)"""
        #domain
        inputs = self._build_inputs(True, self.n_domain)   # domain use always a uniform grid
                                                                # here with n_collocation points
        T_dom = self.AT(inputs)
        CV_dom = self.CV(inputs)

        # store domain datasets
        self.inputs_dom = inputs
        self.T_dom = T_dom
        self.CV_dom = CV_dom


    def _load_dataset(self):
        """load data from a Pykonal experiment"""
        path = os.path.join("data",self.name_example) # directories that store the datasets

        if(self.n_input in [1,2,3]):
            if(self._flag_dataset_build == False):   # load x, y(if n_input>1) and z(if n_input=3)
                x = np.load(os.path.join(path,"x.npy"))[...,None]
                inputs = x
                if(self.n_input > 1):
                    y = np.load(os.path.join(path,"y.npy"))[...,None]
                    inputs = np.concatenate((inputs,y),axis=1)
                if(self.n_input == 3):
                    z = np.load(os.path.join(path,"z.npy"))[...,None]
                    inputs = np.concatenate((inputs,z),axis=1)
                inputs = inputs.astype(np.float32)

                at = np.load(os.path.join(path,"at.npy"))   # load at
                at = at.astype(np.float32)
                v = np.load(os.path.join(path,"v.npy"))     # load v
                v = v.astype(np.float32)

                at = at[...,None]   # from shape (n_coll, ) -> (n_coll, 1)
                if(len(v.shape)==1):
                    v = v[...,None] # add the last dimension only if we are in Isotropic case

                # store domain datasets
                self.inputs_dom = inputs
                self.T_dom = at
                self.CV_dom = v

                # n_domain is simply the length of one of the dataset, x for instance
                self.n_domain = len(x)

                # compute n_collocation using n_domain and prop_collocation (min 10)
                self.n_collocation = max(int(self.n_domain*self.prop_collocation),10)
                # compute n_exact using n_domain and prop_exact (min 3)
                self.n_exact = max(int(self.n_domain*self.prop_exact),3)

                # build collocation dataset from domain dataset
                if(self.n_collocation == self.n_domain): # if n_collocation is equal to n_domain (when prop_collocation = 1.0)
                    self.inputs_coll = inputs
                    self.T_coll = at
                    self.CV_coll = v
                else:
                    # randomly sample n_collocation from the n_domain points
                    index = np.random.randint(low = 0, high=self.n_domain, size=self.n_collocation)
                    self.inputs_coll = inputs[index,:]
                    self.T_coll = at[index,:]
                    self.CV_coll = v[index,:]


                # build exact dataset from domain dataset
                if(self.n_exact == self.n_domain): # if n_exact is equal to n_domain (when prop_exact = 1.0)
                    self.inputs_exact = inputs
                    self.T_exact = at
                    self.CV_exact = v
                else:
                    # randomly sample n_exact from the n_domain points
                    index = np.random.randint(low = 0, high=self.n_domain, size=self.n_exact)
                    self.inputs_exact = inputs[index,:]
                    self.T_exact = at[index,:]
                    self.CV_exact = v[index,:]
                    # x = np.linspace(0.1,0.3,int(self.n_exact/2)) # uniform points
                    # xx = np.linspace(0.7,0.9,int(self.n_exact/2)) # uniform points
                    # x = np.concatenate((x,xx))
                    # inputs = x[...,None]
                    # inputs = inputs.astype(np.float32)
                    # self.inputs_exact = inputs
                    # self.T_exact = inputs#np.sin(10.*inputs)
                    # self.CV_exact = np.zeros_like(inputs)+1.0

                # load original dimensions of the dataset (if provided) (useful only for a better 3D plot...)
                try:
                    path = "data/" + self.name_example + "/dims.npy"
                    dims = np.load(path)
                    n_1 = dims[0]
                    self.n_1 = n_1
                    if(self.n_input > 1):
                        n_2 = dims[1]
                        self.n_2 = n_2
                    if(self.n_input == 3):
                        n_3 = dims[2]
                        self.n_3 = n_3
                except: # if we don't have, we can simply define them like this
                    self.n_1 = self.n_domain
                    self.n_2 = 1
                    self.n_3 = 1

                if(self.pde_type == "anisotropic"):
                    # if anisotropic, load the anisotropic_constraints
                    path = "data/" + self.name_example + "/an_constraints.npy"
                    self.an_constraints = np.load(path)

                # after loading the datasets, we can change the flag to True
                self._flag_dataset_build = True
            else:
                raise Exception("Only case 1D, 2D, 3D allowed")



    def _load_analytical_function(self):
        """load analytical functions"""
        # load the right analytical function according to the dataset name we want to use
        # and store them in self.AT and self.CV
        if(self.name_example == "exponential"):
            self.AT = ATexp
            self.CV = CVexp
        elif(self.name_example == "circle"):
            self.AT = ATcircle
            self.CV = CVcircle
        elif(self.name_example == "anisotropic1"):
            self.AT = ATanis1
            self.CV = CVanis1
        elif(self.name_example == "anisotropic2"):
            self.AT = ATanis2
            self.CV = CVanis2
        else:
            raise Exception("No analytical functions")


    def _build_analytical_dataset(self):
        """build analytical dataset"""
        self._load_analytical_function()    # call all the functions to build the analytical dataset
        self._build_coll_data() # build collocation data
        self._build_exact_data()# build exact data
        self._build_dom_data()  # build domain data

        # build the dimensions
        if(self.n_input == 1):
            self.n_1 = self.n_domain
        if(self.n_input == 2):
            n = math.floor(math.sqrt(self.n_domain))
            self.n_1 = self.n_2 = n
        if(self.n_input == 3):
            n = math.floor(math.sqrt(self.n_domain))
            self.n_1 = self.n_2 = self.n_3 = n

        if(self.pde_type == "anisotropic"):
            # if anisotropic, load the anisotropic_constraints
            path = "data/" + self.name_example + "/an_constraints.npy"
            self.an_constraints = np.load(path)


    def build_dataset(self):
        """
        Build dataset:
        call the functions to build the dataset if analytical or to load
        it if it comes from Pykonal experiment
        """
        if(self._flag_dataset_build == False):  # we build the dataset only the first time
            if(self.dataset_type == "analytical"):
                self._build_analytical_dataset()
            else:
                self._load_dataset()
            self._flag_dataset_build = True

    def build_noisy_dataset(self):
        """
        Add noise to exact data
        """
        if(self._flag_dataset_build == False):   # we build the dataset only the first time
            self.build_dataset()

        if(self._flag_dataset_noise == False):  # we add the noise only the first time
            self.T_with_noise = np.zeros_like(self.T_exact) #store the new exact_data with noise
            self.CV_with_noise = np.zeros_like(self.CV_exact)

            for i in range(0,len(self.T_exact)):
                at_error = np.random.normal(0, self.noise_lv, 1)
                # if we want to weight noise with the magnitude of T_exact
                # at_error = at_error*np.abs(self.T_exact[i])
                self.T_with_noise[i,:] = self.T_exact[i,:] + at_error

                # not so useful... we use onlt T_with_noise, not CV (left it for future different implmentations)
                v_error = np.random.normal(0, self.noise_lv, self.CV_exact.shape[1]) #1, 2 or 3
                self.CV_with_noise[i,:] = self.CV_exact[i,:] + v_error

            self._flag_dataset_noise = True # set the flag to True



    def get_coll_data(self):
        """Return collocation data"""
        if(self._flag_dataset_build == False):   # we build the dataset only the first time
            self.build_dataset()
        return self.inputs_coll,self.T_coll,self.CV_coll


    def get_exact_data(self):
        """ return exact data"""
        if(self._flag_dataset_build == False): # we build the dataset only the first time
            self.build_dataset()
        return self.inputs_exact,self.T_exact,self.CV_exact


    def get_exact_data_with_noise(self):
        """ return exact data + noise """
        if(self._flag_dataset_build == False):  # we build the dataset only the first time
            self.build_dataset()

        if(self._flag_dataset_noise == False):
            self.build_noisy_dataset()

        return self.inputs_exact, self.T_with_noise, self.CV_with_noise


    def get_dom_data(self):
        """ return domain data"""
        if(self._flag_dataset_build == False): # we build the dataset only the first time
            self.build_dataset()
        return self.inputs_dom,self.T_dom,self.CV_dom


    def get_axis_data(self):
        """ return data for axis plot (plot the solution along the line y=x)
        Work only with analytical dataset or for Pykonal datasets only square or cude domain (up to now)"""
        if(self.dataset_type == "analytical"):
            inputs = np.linspace(0,1,100)[...,None]
            for i in range(self.n_input - 1):
                inputs = np.concatenate((inputs, np.linspace(0,1,100)[...,None]),axis=1)
            inputs = inputs.astype(np.float32)
            T = self.AT(inputs)
            CV = self.CV(inputs)
        else:
            n = math.floor(math.sqrt(self.n_domain))+1
            inputs = self.inputs_dom[::n, :]
            T = self.T_dom[::n, :]
            CV = self.CV_dom[::n, :]

        return inputs,T,CV


    def save_dataset(self, save_path):
        """Save the dataset already built in save_path"""
        if(self._flag_dataset_build == False):
            self.build_dataset()

        inputs,at,v = self.get_dom_data()
        final_path_name = os.path.join(save_path, "domain.npz")
        np.savez(final_path_name, inputs=inputs, at=at, v=v)

        inputs,at,v = self.get_coll_data()
        final_path_name = os.path.join(save_path, "collocation.npz")
        np.savez(final_path_name, inputs=inputs, at=at, v=v)

        inputs,at,v = self.get_exact_data()
        final_path_name = os.path.join(save_path, "exact.npz")
        np.savez(final_path_name, inputs=inputs, at=at, v=v)


    def plot(self, save_path = "plot"):
        """Plot the exact experiment"""
        if(self._flag_dataset_build == False):
            self.build_dataset()
        if(self._flag_dataset_noise == False):
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
        x,u,f = self.get_dom_data()
        x_e,u_e,f_e = self.get_exact_data()
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
        inputs,at,v = self.get_dom_data()
        inputs_e,at_e,v_e = self.get_exact_data()

        x = inputs[:,0]
        y = inputs[:,1]
        x_e = inputs_e[:,0]
        y_e = inputs_e[:,1]

        fig = plt.figure()
        fig.set_size_inches((12,5))
        plt.subplot(1,2,1)
        plt.scatter(x, y, c=at, label = 'at', cmap = 'coolwarm', vmin = min(at), vmax = max(at))
        plt.colorbar()

        xx = x.reshape((self.n_1,self.n_2))
        yy = y.reshape((self.n_1,self.n_2))
        tt = at.reshape((self.n_1,self.n_2))

        plt.contour(xx, yy, tt, levels=20, colors="black", alpha=0.5)
        plt.scatter(x_e, y_e, facecolors = 'none', edgecolor = 'k')
        plt.axis("equal")
        plt.xlim([0,np.max(x)])
        plt.ylim([0,np.max(y)])

        plt.subplot(1,2,2)
        plt.scatter(x, y, c=v, label = 'v', cmap = 'coolwarm', vmin = min(v), vmax = max(v))
        plt.colorbar()
        plt.axis("equal")
        plt.xlim([0,np.max(x)])
        plt.ylim([0,np.max(y)])

        final_path_name = os.path.join(save_path, "exact.png")
        plt.savefig(final_path_name)


    def _plot_3D(self, save_path):
        """plot the 3D example"""
        inputs,at,v = self.get_dom_data()
        #inputs_e,at_e,v_e = self.get_exact_data()

        x = inputs[:,0]
        y = inputs[:,1]
        z = inputs[:,2]


        xx = np.reshape( x, (self.n_1,self.n_2,self.n_3))
        yy = np.reshape( y, (self.n_1,self.n_2,self.n_3))
        zz = np.reshape( z, (self.n_1,self.n_2,self.n_3))
        tt = np.reshape( at, (self.n_1,self.n_2,self.n_3))
        vv = np.reshape( v, (self.n_1,self.n_2,self.n_3))

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
            inputs,_,_ = self.datasets_class.get_dom_data()
            x = inputs[:,0]
            y = inputs[:,1]
            plt.figure()
            plt.scatter(x,y)
            plt.axis('equal')
            #plt.show()
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
        else:
            pass


    def get_num_collocation(self):
        """get number of collocation points"""
        if(self._flag_dataset_build == False):
            self.build_dataset()
        return self.n_collocation


    def get_num_exact(self):
        """get number of exact points"""
        if(self._flag_dataset_build == False):
            self.build_dataset()
        return self.n_exact


    def get_n_input(self):
        """get n input"""
        return self.n_input
