
import random

DEFAULT_CAPACITY = 100000

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
		self.memory[self.last_index] = sars_tuple	
		if (self.last_index + 1 - self.first_index) > self.capacity:
			del self.memory[self.first_index]
			self.first_index += 1
			#print "discarding sample: " + str(self.first_index - 1)

	def sample(self): # returns SARS' tuple
		if (self.first_index == -1):
			return
		if (self.capacity > self.last_index + 1 - self.first_index:
			return self.memory[last_index] # dont sample randomly until replay is full
		rand_sample_index = random.randint(self.first_index, self.last_index)
		#print "num in memory: " + str(len(self.memory)) + " rand index: " + str(rand_sample_index)
		return self.memory[rand_sample_index]
