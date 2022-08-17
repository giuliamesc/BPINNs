import numpy as np
values_npy = np.array([
    [1,2,3,4,5],
    [6,7,8,9,10],
    [11,12,13,14,15],
    [16,17,18,19,20]
    ])

keys = ["A","B", "C", "D"]
value_dict = {
            keys[0] : list(values_npy[0,:]), 
            keys[1] : list(values_npy[1,:]),
            keys[2] : list(values_npy[2,:]),
            keys[3] : list(values_npy[3,:]),
        }
#print(value_dict)

print(values_npy)
values_npy[3,:] = [1,2,3,4,5]
print(values_npy)