"""
:description: save and load weight vectors
"""

import pickle

def save_weights(weights):
	output_filename = '/Users/wulfe/Dropbox/School/Stanford/autumn_2015/cs221/project/scripts/weights/weights.pkl'
	with open(output_filename, 'wb') as f:
		pickle.dump(weights, f)

def load_weights():
	input_filename = '/Users/wulfe/Dropbox/School/Stanford/autumn_2015/cs221/project/scripts/weights/weights.pkl'
	with open(input_filename, 'rb') as f:
		return pickle.load(f)
