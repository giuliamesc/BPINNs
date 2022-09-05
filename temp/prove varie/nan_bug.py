import math
import numpy as np
import tensorflow as tf

a = 1.5
b = math.nan

alpha = min(0, -b+a)
alpha = min(0, a-b)
alpha = float('-inf')

print(f"a: {a :1.2f}")
print(f"b: {b :1.2f}")
print(f"alpha: {np.exp(alpha)*100 :1.2f}%")

t = 1e-2
print("t:", t)
print("t:", tf.cast(t, dtype="float32"))