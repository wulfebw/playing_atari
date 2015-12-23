"""
:description: save and load weight vectors and theano models
"""
import os
import csv
import pickle
import cPickle
import datetime

import numpy as np

WEIGHTS_DIR = "../weights"
REWARDS_DIR = "../rewards"
MODELS_DIR = "../models"

def save_weights(weights, filename):
    output_filepath = os.path.join(WEIGHTS_DIR, "{}.pkl".format(filename))
    with open(output_filepath, 'wb') as f:
        pickle.dump(weights, f)

def load_weights(filename):
    print "loading weights..."
    filepath = os.path.join(WEIGHTS_DIR, filename)
    weights = {}
    try:
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
    except IOError as e:
        print("!!!weight file {} not found, reinitializing weights!!!".format(filepath))
        raise(e)
    return weights

def save_model(model, model_filename):
    f = file(os.path.join(MODELS_DIR, model_filename), 'wb')
    try:
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    except:
        print('could not save model')
    f.close()

def load_model(model_filename):
    f = file(os.path.join(MODELS_DIR, model_filename), 'rb')
    model = cPickle.load(f)
    f.close()
    return model

def save_rewards(rewards, filename):
    filepath = os.path.join(REWARDS_DIR, "{}".format(filename))
    np.savez(filepath, rewards=rewards)

def load_rewards(filename):
    filepath = os.path.join(REWARDS_DIR, "{}".format(filename))
    return np.load(filepath)['rewards']
