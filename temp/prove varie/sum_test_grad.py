import tensorflow as tf
import numpy as np

x = np.array([[1]])
x = tf.convert_to_tensor(x, dtype="float32")
#u = sum(l)

with tf.GradientTape() as tape:
    tape.watch(x)
    l = [1*x,2*x,3*x]
    u = sum(l)
print(tape.jacobian(l,x))