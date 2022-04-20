import tensorflow as tf

class dataloader:
	"""
	Class to handle the dataloader for mini-batch training (for collocation points)
	It can be used also for mini-batch training with exact data if needed
	"""
	def __init__(self, datasets_class, batch_size, reshuffle_every_epoch):
		"""!
		Constructor

		@param datasets_class an object of type datasets_class that contains all the datasets we need
		@param batch_size dimension of a batch_size for collocation points
		@param reshuffle_every_epoch boolean that indicates if we want to reshuffle the points at every epoch
		"""
		## datasets_class object, we'll use its attributes (coll_data, num_collocation ...)
		self.datasets_class = datasets_class
		## batch size of collocation points for minibatch training
		self.batch_size = batch_size
		## boolean for reshuffle_every_epoch or not
		self.reshuffle_each_iteration = reshuffle_every_epoch

	def dataload_collocation(self):
		"""Return a dataloader for collocation points.
		Implemented using tf.data.Dataset.from_tensor_slices and batch"""
		# get the collocation data
		inputs, _, _ = self.datasets_class.coll_data

		#load the data of collocation
		data = tf.data.Dataset.from_tensor_slices(inputs)

		#load data in batch of size = batch_size and shuffle
		data = data.shuffle(buffer_size=(self.datasets_class.num_collocation+1),
							reshuffle_each_iteration=self.reshuffle_each_iteration)

        # if we set self.batch_size = 0, we are asking to NOT use a mini-batch implementation
		if(self.batch_size == 0):
			self.batch_size = self.datasets_class.num_collocation
		else:
			if(self.batch_size > self.datasets_class.num_collocation):
				print("batch size:", self.batch_size)
				print( "coll size:", self.datasets_class.num_collocation)
				raise Exception("Batch size can't be bigger than actual collocation size!")

		coll_loader = data.batch(batch_size=self.batch_size)

		return coll_loader, self.batch_size

	def dataload_exact(self, exact_batch_size):
		"""!
		Return a dataloader for exact points (only if needed.)
		You have to provide an additional parameter "exact_batch_size"
		Implemented using tf.data.Dataset.from_tensor_slices and batch

		@param exact_batch_size dimension of exact points batch"""
		# get the exact (noisy) data
		inputs, _, _ = self.datasets_class.exact_data_noise

		#load the data of collocation
		data = tf.data.Dataset.from_tensor_slices(inputs)

		#load data in batch of size = batch_size and shuffle
		data = data.shuffle(buffer_size=(self.datasets_class.num_exact+1),
							reshuffle_each_iteration=self.reshuffle_each_iteration)
		exact_loader = data.batch(batch_size=exact_batch_size)

		return exact_loader, exact_batch_size
