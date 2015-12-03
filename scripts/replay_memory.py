
import random

DEFAULT_CAPACITY = 10000

class ReplayMemory:
	def __init__(self, capacity=DEFAULT_CAPACITY):
		self.memory = {}
		self.first_index = -1
		self.last_index = -1
		self.capacity = capacity

	def store(self, sars_tuple): # stores SARS' tuple
		if (self.first_index == -1):
			self.first_index = 0
		self.last_index += 1
		self.memory[(last_index + 1)] = sars_tuple	
		if (self.last_index + 1 - self.first_index) > capacity:
			del self.memory[self.first_index]
			self.first_index += 1

	def sample(self): # returns SARS' tuple
		if (self.first_index == -1):
			return
		rand_sample_index = random.randint(self.first_index, self.last_index)
		return self.memory[rand_sample_index]
