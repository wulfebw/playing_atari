"""
:description: save and load weight vectors and theano models
"""

import os
import pickle
import cPickle
import datetime

def save_weights(weights, episodic=False):

	output_filename = "weights/{}.pkl".format(datetime.datetime.now().isoformat())
	if episodic:
		output_filename = "weights/episodic-{}.pkl".format(datetime.datetime.now().isoformat())
	with open(output_filename, 'wb') as f:
		pickle.dump(weights, f)

def load_weights():
	weight_files = sorted(os.listdir('weights'), reverse=True)
	if weight_files:
		input_filename = os.path.join('weights', weight_files[0])
	else:
		return None
	print input_filename

	weights = None
	try:
		with open(input_filename, 'rb') as f:
			weights = pickle.load(f)
	except IOError as e:
		print("weight file {} not found, reinitializing weights".format(input_filename))
	return weights

def save_model(model, output_filename):
	f = file(output_filename, 'wb')
	cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()

def load_model(model_filename):
	f = file(model_filename, 'rb')
	model = cPickle.load(f)
	f.close()
	return model
