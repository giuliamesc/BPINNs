from oper_class import *
import tensorflow as tf
import numpy as np

def print_tensor(tensor, name= None):
    if name is not None:
        print(f"\nTensor: {name}")
    print(f"Shape: {tensor.shape}")
    print(tensor.numpy())

n_sample = 5
dim_inp  = 2
dim_out  = 3

a = np.linspace(1, n_sample, n_sample)[...,None]
b = np.linspace(0, n_sample-1, n_sample)[...,None]
c = np.linspace(1, n_sample, n_sample)[...,None]

x = np.concatenate([a,b,c][0:dim_inp], axis=-1)
x = tf.cast(x, dtype = "float32")
print_tensor(x, "input")


def model(inputs):
    l1 = tf.keras.layers.Dense(4, activation=tf.nn.sigmoid)(inputs)
    l2 = tf.keras.layers.Dense(4, activation=tf.nn.sigmoid)(l1)
    outputs = tf.keras.layers.Dense(dim_out)(l2) 
    return outputs

def func(inputs):
    x = inputs[:,0:1]
    y = inputs[:,1:2]
    a = 2*x*x*y
    b = x*x
    c = 3*y*y
    return tf.stack([a[:,0],b[:,0],c[:,0]][0:dim_out], axis=-1)

with tf.GradientTape(persistent = True) as tape:
    tape.watch(x)
    outputs = tf_unpack(func(x))
    u = outputs[0]
    #f = laplacian_scalar(tape, u, x)
    f = laplacian_vector(tape, outputs, x)
print(f)