class A():
    def print_child(self):
        print(type(self).__name__)

class B(A): pass
class C(A): pass
class D(A): pass

b = B()
b = C()
b.print_child()

import tensorflow as tf
import numpy as np

t = np.array([[5,2,0],[1,3,4]])
t = tf.convert_to_tensor(t, dtype="float32")

res = tf.norm(t)
print(res*res)