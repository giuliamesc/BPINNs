import numpy as np 

class BatcherIndex:
    
    def __init__(self, num, bs):
        self.indexes = np.arange(num)
        self.num_items  = num
        self.batch_size = bs
        self.ptr = 0
        np.random.shuffle(self.indexes)
    
    def __next__(self):
        if self.ptr + self.batch_size > self.num_items:
            np.random.shuffle(self.indexes)
            self.ptr = 0
        result = self.indexes[self.ptr: self.ptr+self.batch_size]
        self.ptr += self.batch_size
        return result

class Batcher:

    def __init__(self, dic, num, bs):
        self.values  = dic
        self.indexes = BatcherIndex(num, bs)

    def __next__(self):
        idx = next(self.indexes)
        return {k:v[idx] for k,v in self.values.items()}

class BatcherDataset:
    
    def __init__(self, dataset, num_batch=1):
        self.dataset = dataset
        self.num_batch = num_batch
        self.__build_batcher()
        
    def __build_batcher(self):
        nums = self.dataset.num_points
        batch_size = lambda n: max(1, n//self.num_batch)
        self.__data_sol_batcher = Batcher(self.dataset.data_sol, nums["sol"], batch_size(nums["sol"]))
        self.__data_par_batcher = Batcher(self.dataset.data_par, nums["par"], batch_size(nums["par"]))
        self.__data_bnd_batcher = Batcher(self.dataset.data_bnd, nums["bnd"], batch_size(nums["bnd"]))
        self.__data_pde_batcher = Batcher(self.dataset.data_pde, nums["pde"], batch_size(nums["pde"]))

    def __next__(self):
        self.__data_sol = next(self.__data_sol_batcher)
        self.__data_par = next(self.__data_par_batcher)
        self.__data_bnd = next(self.__data_bnd_batcher)
        self.__data_pde = next(self.__data_pde_batcher)

    @property
    def data_sol(self):
        return self.__data_sol
    @property
    def data_par(self):
        return self.__data_par
    @property
    def data_bnd(self):
        return self.__data_bnd
    @property
    def data_pde(self):
        return self.__data_pde