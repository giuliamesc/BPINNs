import numpy as np
import os

folder_name = "outs/Regression/reg1D_cos"
case_name  = "trash"
full_path  = os.path.join(folder_name, case_name)
theta_path = os.path.join(full_path, "thetas")

for f in os.listdir(os.listdir(theta_path)[-1]):
    theta = np.load(f)