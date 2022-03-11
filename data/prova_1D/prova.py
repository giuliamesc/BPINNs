import torch
import hamiltorch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

x = np.load("x.npy")
x1 = x[100:300][::4]
x2 = x[700:900][::4]
x = np.concatenate((x1,x2))
x = x[...,None]
x = x.reshape((100,1))
x = x.astype(np.float32)
x = torch.tensor(x)
y = np.load("at.npy")
y1 = y[100:300][::4]
y2 = y[700:900][::4]
y = np.concatenate((y1,y2))
y = y[...,None]
y = y.reshape((100,1))

noise_level = 0.01
for i in range(y.shape[0]):
    y_error = np.random.normal(0, noise_level, 1)
    y[i,0] += y_error

y = y.astype(np.float32)
y = torch.tensor(y)

#breakpoint()

hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Net(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

step_size = 0.0005
num_samples = 10000
L = 30 # Remember, this is the trajectory length
burn = -1
store_on_GPU = False # This tells sampler whether to store all samples on the GPU
debug = False # This is useful for debugging if we want to print the Hamiltonian
model_loss = 'regression'
tau = torch.tensor([1., 1., 1., 1., 1., 1.], device=device)
tau_out = 1000.0
#tau_out = 110.4
tau_list = torch.tensor([1., 1., 1., 1., 1., 1.], device=device)
mass = 1.0 # Mass matrix diagonal scale
params_init = hamiltorch.util.flatten(net).to(device).clone()
inv_mass = torch.ones(params_init.shape) / mass # Diagonal of inverse mass matrix

print(params_init.shape)
integrator = hamiltorch.Integrator.EXPLICIT
sampler = hamiltorch.Sampler.HMC # We are doing simple HMC with a standard leapfrog

hamiltorch.set_random_seed(0)
# Let's sample!
params_hmc_f = hamiltorch.sample_model(net, x, y, params_init=params_init,
                                       model_loss=model_loss, num_samples=num_samples,
                                       inv_mass=inv_mass.to(device), step_size=step_size,
                                       num_steps_per_sample=L ,tau_out=tau_out, tau_list=tau_list,
                                       store_on_GPU=store_on_GPU, sampler = sampler)


x_test = np.load("x.npy")
x_test = x_test.reshape((1000,1))
x_test = x_test.astype(np.float32)
x_val = torch.tensor(x_test)
y_test = np.load("at.npy")
y_test = y_test.reshape((1000,1))
y_test = y_test.astype(np.float32)
y_val = torch.tensor(y_test)

pred_list, log_prob_list = hamiltorch.predict_model(net, x = x_val,
                                                  y = y_val, samples=params_hmc_f,
                                                  model_loss=model_loss, tau_out=tau_out,
                                                  tau_list=tau_list)

burn = 3000

xx = np.load("x.npy")
xx = xx.reshape((1000,1))
xx = xx.astype(np.float32)
xx = torch.tensor(xx)
yy = np.load("at.npy")
yy = yy.reshape((1000,1))
yy = yy.astype(np.float32)
yy = torch.tensor(yy)


plt.figure(figsize=(10,5))
plt.plot(x_val.cpu().numpy(),pred_list[burn:].cpu().numpy().squeeze().T, 'C0',alpha=0.051)
plt.plot(x_val.cpu().numpy(),pred_list[burn:].mean(0).cpu().numpy().squeeze().T, 'C1',alpha=0.9)
plt.plot(x_val.cpu().numpy(),pred_list[burn:].mean(0).cpu().numpy().squeeze().T +pred_list.std(0).cpu().numpy().squeeze().T, 'C1',alpha=0.8,linewidth=3)
plt.plot(x_val.cpu().numpy(),pred_list[burn:].mean(0).cpu().numpy().squeeze().T -pred_list.std(0).cpu().numpy().squeeze().T, 'C1',alpha=0.8,linewidth=3)

plt.plot(x.cpu().numpy(),y.cpu().numpy(),'.C3',markersize=30, label='x train',alpha=0.6)
plt.plot(xx.cpu().numpy(),yy.cpu().numpy(),'r-',label='true',alpha=0.6)

plt.legend(fontsize=20)
plt.show()
plt.savefig("010.png")
