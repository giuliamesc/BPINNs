import numpy as np
import tensorflow as tf

outputs = tf.random.normal(shape=[5,2])
targets = np.random.randn(5,2)

norm = tf.norm(outputs - targets, axis = -1)
mse  = tf.keras.losses.MSE(norm, tf.zeros_like(norm))
print(mse.shape)
print(mse)