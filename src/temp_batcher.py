import numpy as np 

class BatcherIndex:
  def __init__(self, num, bs): 
    self.indices = np.arange(num)
    self.num_items  = num
    self.batch_size = bs
    self.ptr = 0
    np.random.shuffle(self.indices)
  def __next__(self):
    if self.ptr + self.batch_size > self.num_items:
      np.random.shuffle(self.indices)
      self.ptr = 0
    result = self.indices[self.ptr:self.ptr+self.batch_size]
    self.ptr += self.batch_size
    return result

class Batcher:
  def __init__(self, dic, num, bs): 
    self.values  = dic
    self.indices = BatcherIndex(num, bs) 
  def __next__(self):
    idx = next(self.indices)
    return {k:v[idx] for k,v in self.values.items()}

num = 12
bs  = 5

vec_1 = 2*np.arange(num) + 20
vec_2 = vec_1 + 3
dic = {"1": vec_1, "2": vec_2}

batcher = Batcher(dic, num, bs)
for ep in range(10):
  print(next(batcher))