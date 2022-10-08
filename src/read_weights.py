import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm

folder_name = "outs/Regression/reg1D_cos"
case_name  = "trash"
full_path  = os.path.join(folder_name, case_name)
thetas_path = os.path.join(full_path, "thetas")
theta_path = os.path.join(thetas_path, os.listdir(thetas_path)[-1])
theta = list()

for f in os.listdir(theta_path):
    fp = os.path.join(theta_path,f)
    theta.append(np.load(fp))

b1 = np.reshape(theta[0], newshape=[np.size(theta[0]),1])
b2 = np.reshape(theta[1], newshape=[np.size(theta[1]),1])
b3 = np.reshape(theta[2], newshape=[np.size(theta[2]),1])

w1 = np.reshape(theta[3], newshape=[np.size(theta[3]),1])
w2 = np.reshape(theta[4], newshape=[np.size(theta[4]),1])
w3 = np.reshape(theta[5], newshape=[np.size(theta[5]),1])

w_list = [w1,w2,w3]
titles = ["w1","w2","w3"]

"""
for w,t in zip(w_list,titles):    
    print(f"Avg of {t}: \t {np.mean(w)}")
    print(f"Std of {t}: \t {np.std(w)}")

    x = np.arange(np.min(w), np.max(w), 0.1)

    plt.figure()

    plt.hist(w, density=True, bins=10, edgecolor='k')
    plt.plot(x, norm.pdf(x, 0., 1.), linewidth = 5)
    plt.title("Histogram of "+ t)
"""

theta = np.concatenate([w1,w2,w3,b1,b2,b3])
print(f"Avg of all thetas: \t {np.mean(theta)}")
print(f"Std of all thetas: \t {np.std(theta)}")

x = np.arange(np.min(theta), np.max(theta), 0.1)

plt.figure()

plt.hist(theta, density=True, bins=10, edgecolor='k')
plt.plot(x, norm.pdf(x, 0., 1.), linewidth = 5)
plt.title("Histogram of all thetas")

plt.show()