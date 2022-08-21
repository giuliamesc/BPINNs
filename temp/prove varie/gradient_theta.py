import tensorflow as tf
import numpy as np
tf.random.set_seed(21)

n_input  = 1
n_neuron = 2
n_output = 1

x_0 = np.array([10])
x_0 = tf.convert_to_tensor(x_0)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(n_input,)))
model.add(tf.keras.layers.Dense(n_neuron, activation="linear", kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(tf.keras.layers.Dense(n_output, activation="linear", kernel_initializer='glorot_uniform', bias_initializer='zeros'))

with tf.GradientTape(persistent=True) as tape:
    tape.watch(model.trainable_variables)
    u = model(x_0)
u_w = tape.gradient(u, model.trainable_variables)

for tv, uw in zip(model.trainable_variables, u_w):
    print("\n")
    print(tv)
    print(uw)