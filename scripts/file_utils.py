"""
:description: save and load weight vectors
"""

import pickle

def save_weights(weights):
	output_filename = "weights/w.pkl"
	with open(output_filename, 'wb') as f:
		pickle.dump(weights, f)

def load_weights():
	input_filename = "weights/w.pkl"
	weights = None
	try:
		with open(input_filename, 'rb') as f:
			weights = pickle.load(f)
	except IOError as e:
		print("weight file {} not found, reinitializing weights".format(input_filename))
		weights = None
	return weights
