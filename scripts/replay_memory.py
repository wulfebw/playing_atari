
import random

DEFAULT_CAPACITY = 10000

class ReplayMemory(object):
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
			self.discardSample(True)

	def isFull(self):
		return self.last_index + 1 - self.first_index >= self.capacity

	def discardSample(self, bias=False):
		biasChance = 0.9
		while True:
			rand_index = random.randint(self.first_index, self.last_index)
			tuple_d = self.memory[rand_index]
			if (abs(tuple_d[2]) > 0) and bias and random.random() < biasChance:
				continue
			first_tuple = self.memory[self.first_index]
			del self.memory[rand_index]
			if rand_index == self.first_index:
				break
			del self.memory[self.first_index]
			self.memory[rand_index] = first_tuple
			break
		self.first_index += 1

	def sample(self): # returns SARS' tuple
		if (self.first_index == -1):
			return
		if self.capacity > self.last_index + 1 - self.first_index:
			return self.memory[self.last_index] # dont sample randomly until replay is full
		rand_sample_index = random.randint(self.first_index, self.last_index)
		return self.memory[rand_sample_index]
