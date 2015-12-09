"""
:description: train an agent to play a game
"""

import os
import sys
import copy
import random

import numpy as np
import theano
import theano.tensor as T
from sklearn import preprocessing
from sklearn.decomposition import PCA

# atari learning environment imports
from ale_python_interface import ALEInterface

# custom imports
import file_utils
import screen_utils
import feature_extractors
import learning_agents
from replay_memory import ReplayMemory
from mlp import MLP, HiddenLayer, OutputLayer

# the input size of the network
MAX_FEATURES = 8

def train(gamepath, 
          n_episodes=10000, 
          display_screen=False, 
          record_weights=True, 
          reduce_exploration_prob_amount=0.00001,
          n_frames_to_skip=4,
          exploration_prob=.3,
          verbose=True,
          discount=.995,
          learning_rate=.01,
          load_weights=False,
          frozen_target_update_period=5,
          use_replay_mem=True):
    """
    :description: trains an agent to play a game 

    :type gamepath: string 
    :param gamepath: path to the binary of the game to be played

    :type n_episodes: int 
    :param n_episodes: number of episodes of the game on which to train

    display_screen : whether or not to display the screen of the game 
    
    record_weights : whether or not to save the weights of the nextwork
    
    reduce_exploration_prob_amount : amount to reduce exploration prob each episode
                                     to not reduce exploration_prob set to 0
    
    n_frames_to_skip : how frequently to determine a new action to use
    
    exploration_prob : probability of choosing a random action
    
    verbose : whether or not to print information about the run periodically
    
    discount : discount factor used in learning 
    
    learning_rate : the scaling factor for the sgd update
    
    load_weights : whether or not to load weights for the network (set the files directly below)
    
    frozen_target_update_period : the number of episodes between reseting the target of the network
    """

    # load the ale interface to interact with
    ale = ALEInterface()
    ale.setInt('random_seed', 42)

    # display/recording settings, doesn't seem to work currently
    recordings_dir = './recordings/breakout/'
    # previously "USE_SDL"
    if display_screen:
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            ale.setBool('sound', False) # Sound doesn't work on OSX
            #ale.setString("record_screen_dir", recordings_dir);
        elif sys.platform.startswith('linux'):
            ale.setBool('sound', True)
        ale.setBool('display_screen', True)
    ale.loadROM(gamepath)
    real_actions = [3,4] # ale.getMinimalActionSet()
    actions = np.arange(len(real_actions))

    # these theano variables are used to define the symbolic input of the network
    features = T.dvector('features')
    action = T.lscalar('action')
    reward = T.dscalar('reward')
    next_features = T.dvector('next_features')

    # load weights by file name
    # currently must be loaded by individual hidden layers
    if load_weights:
        hidden_layer_1 = file_utils.load_model('weights/hidden0_replay.pkl')
        hidden_layer_2 = file_utils.load_model('weights/hidden1_replay.pkl')
    else:
        # defining the hidden layer network structure
        # the n_hid of a prior layer must equal the n_vis of a subsequent layer
        # for q-learning the output layer must be of len(actions)
        init_W_fake_1 = np.array([[1],[-1]])
        hidden_layer_1 = HiddenLayer(n_vis=2, n_hid=1, layer_name='hidden1', activation='relu', init_W_fake=init_W_fake_1)
        init_W_fake_2 = np.array([[-1, 1]])
        hidden_layer_2 = HiddenLayer(n_vis=1, n_hid=2, layer_name='hidden2', activation='relu', init_W_fake=init_W_fake_2)
        #hidden_layer_3 = HiddenLayer(n_vis=MAX_FEATURES, n_hid=len(actions), layer_name='hidden3', activation='relu') 
    output_layer = OutputLayer(layer_name='output', activation='relu')

    layers = [hidden_layer_1, hidden_layer_2]
    mlp = MLP(layers, discount=discount, learning_rate=learning_rate)

    # sym_action = mlp.get_action(features)
    # get_action = theano.function([features], sym_action)

    rewards = []
    best_reward = 4

    preprocessor = screen_utils.RGBScreenPreprocessor()
    feature_extractor = feature_extractors.TrackingClassifyingContourExtractor(max_features=MAX_FEATURES)
    old_err = 0
    KP = 1
    KD = 3
    for episode in xrange(n_episodes):
        total_reward = 0
        action = 1
        counter = 0
        reward = 0
        lives = ale.lives()
        screen = np.zeros((32,32,3))
        state = { "screen" : screen, "objects" : None, "prev_objects": None, "features": np.zeros(2)}
        while not ale.game_over():

            counter += 1
            if counter < 10: 
                ale.act(1)
                continue
            if ale.lives() < lives: 
                lives = ale.lives()
                for _ in xrange(10):
                    ale.act(1)
                continue

            features = state["features"]
            #action = get_action(features)
            paddle_x = features[0]
            ball_x = features[1]
            err = ball_x - paddle_x

            control = KP*err + KD *(err-old_err)
            #print err,(err-old_err),control
            old_err = err
            if control > 0:
                action = 0
                #print 'right'
            else:
                action = 1
                #print 'left'
            # try:
            #     action = int(raw_input())
            # except:
            #     action = 1
            # if action != 0 and action != 1:
            #     action = 1
            # print action

            reward = ale.act(real_actions[action])
            next_screen = ale.getScreenRGB()
            next_screen = preprocessor.preprocess(next_screen)
            next_state = {"screen": next_screen, "objects": None, "prev_objects": state["objects"]}
            next_features = feature_extractor(next_state, action=None)
            
            #print next_features

            paddle_x = 0
            ball_x = 0
            for tup in next_features:
                if tup[-1] != 0 or tup[-2] != 0:
                    pass
                if tup[3] == 237:
                    paddle_x = tup[2]
                if tup[-1] !=0:
                    ball_x = tup[2]
            next_features = [paddle_x, ball_x]

            next_state["features"] = next_features
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        print('episode: {} ended with score: {}'.format(episode, rewards[-1]))
        ale.reset_game()
    return rewards

if __name__ == '__main__':
    base_dir = 'roms'
    game = 'breakout.bin'
    gamepath = os.path.join(base_dir, game)
    rewards = train(gamepath, 
                    n_episodes=10000, 
                    display_screen=False, 
                    record_weights=False, 
                    reduce_exploration_prob_amount=0.0002,
                    n_frames_to_skip=4,
                    exploration_prob=0.0,
                    verbose=True,
                    discount=0.99,
                    learning_rate=.0004,
                    load_weights=False,
                    frozen_target_update_period=5,
                    use_replay_mem=False)
