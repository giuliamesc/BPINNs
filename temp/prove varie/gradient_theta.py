import tensorflow as tf
import numpy as np
tf.random.set_seed(21)

n_input  = 1
n_neuron = 2
n_output = 2

#x_0 = np.array([[1]])
x_0 = np.array([[1],[2],[3]])
x_0 = tf.convert_to_tensor(x_0)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(n_input,)))
model.add(tf.keras.layers.Dense(n_neuron, activation="linear", kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(tf.keras.layers.Dense(n_output, activation="linear", kernel_initializer='glorot_uniform', bias_initializer='zeros'))

def shape_list(l):
    return [t.shape for t in l]

def __mse(vect):
    norm = tf.norm(vect, axis = -1)
    return tf.keras.losses.MSE(norm, tf.zeros_like(norm))

def __normal_loglikelihood(mse, n, log_var):
    return -1*(- 0.5 * n * mse * tf.math.exp(log_var) + 0.5 * n * log_var)

def __convert(tensor):    
    return tf.cast(tensor, dtype=tf.float32)

def func(outputs, n_d=3, log_var=1.):
    mse_data = __mse(outputs)
    log_data = __normal_loglikelihood(mse_data, n_d, log_var)
    return __convert(mse_data), __convert(log_data)

with tf.GradientTape() as tape1:
    tape1.watch(model.trainable_variables)
    u = model(x_0)
u_w = tape1.jacobian(u, model.trainable_variables)

with tf.GradientTape() as tape2:
    tape2.watch(u)
    l = func(u)
l_u = tape2.gradient(l, u)

with tf.GradientTape() as tape3:
    tape3.watch(model.trainable_variables)
    u = model(x_0)
    l = func(u)
l_w = tape3.gradient(l, model.trainable_variables)

#print(shape_list(model.trainable_variables))
#print(shape_list(u_w))
#print(l_u.shape)
print(shape_list(l_w))

l_w_custom = list()
for k in range(len(u_w)):
    res = 0
    for i in range(3):
        for j in range(n_output):
            res += u_w[k][i][j]*l_u[i][j] 
    l_w_custom.append(res)
print(shape_list(l_w_custom))


for i,j in zip(l_w, l_w_custom):
    print("\n")
    print(i)
    print(j)