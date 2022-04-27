import tensorflow as tf

class dataloader:
	"""
	Class to handle the dataloader for mini-batch training (for collocation points)
	It can be used also for mini-batch training with exact data if needed
	"""
	def __init__(self, datasets_class, batch_size, random_seed, reshuffle=True):
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
		## seed for shuffling
		self.seed = random_seed
		## boolean for reshuffle_every_epoch or not
		self.reshuffle = reshuffle

	def dataload_collocation(self):
		"""Return a dataloader for collocation points.
		Implemented using tf.data.Dataset.from_tensor_slices and batch"""

		# if we set self.batch_size = 0, we are asking to NOT use a mini-batch implementation
		if not self.batch_size:
			self.batch_size = self.datasets_class.num_collocation
		if self.batch_size > self.datasets_class.num_collocation:
			print("Batch size:", self.batch_size)
			print( "Coll size:", self.datasets_class.num_collocation)
			raise Exception("Batch size can't be bigger than actual collocation size!")

		# get the collocation data
		inputs, _, _ = self.datasets_class.coll_data
		#load the data of collocation
		data = tf.data.Dataset.from_tensor_slices(inputs)
		
		#load data in batch of size = batch_size and shuffle
		data = data.shuffle(buffer_size = data.cardinality().numpy(), 
							seed = self.seed, 
							reshuffle_each_iteration = self.reshuffle)
		coll_loader = data.batch(batch_size=self.batch_size, drop_remainder=True)

		return coll_loader

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
							reshuffle_each_iteration=self.reshuffle)
		exact_loader = data.batch(batch_size=exact_batch_size)

		return exact_loader
