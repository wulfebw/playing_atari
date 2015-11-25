"""
:description: save and load weight vectors
"""

import os
import pickle
import datetime

def save_weights(weights):

	output_filename = "weights/{}.pkl".format(datetime.datetime.now().isoformat())
	with open(output_filename, 'wb') as f:
		pickle.dump(weights, f)

def load_weights():
	print "loading weights..."
	use_latest_weights = False
	if (use_latest_weights):
		weight_files = sorted(os.listdir('weights'), reverse=True)
		if weight_files:
			input_filename = os.path.join('weights', weight_files[0])
		else:
			return None
	else:
		input_filename = os.path.join('weights', '1bucket-newest2.pkl')
	print "Loading weight from file: " + input_filename
	weights = None
	try:
		with open(input_filename, 'rb') as f:
			weights = pickle.load(f)
	except IOError as e:
		print("weight file {} not found, reinitializing weights".format(input_filename))
	return weights
