"""
:description: save and load weight vectors and theano models
"""
import os
import csv
import pickle
import cPickle
import datetime

import numpy as np

import mlp

def save_weights(weights, filename):
    output_filepath = "weights/{}.pkl".format(filename)
    with open(output_filename, 'wb') as f:
        pickle.dump(weights, f)

def load_weights(filename):
    print "loading weights..."
    filepath = os.path.join('weights', filename)
    weights = {}
    try:
        with open(input_filename, 'rb') as f:
            weights = pickle.load(f)
    except IOError as e:
        print("weight file {} not found, reinitializing weights".format(input_filename))
    return {}

def save_model(model, model_filename):
    f = file(model_filename, 'wb')
    try:
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    except:
        print('could not save model')
    f.close()

def load_model(model_filename):
    f = file(model_filename, 'rb')
    model = cPickle.load(f)
    f.close()
    return model

def save_rewards(rewards, output_filename='latest'):
    output_filepath = "rewards/{}".format(output_filename)
    np.savez(output_filepath, rewards=rewards)

def load_rewards(input_filename='latest.npz'):
    input_filepath = "rewards/{}".format(input_filename)
    return np.load(input_filepath)['rewards']
