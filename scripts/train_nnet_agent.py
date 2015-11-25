"""
:description: train an agent to play a game
"""

import os
import sys
import random

import numpy as np
import theano
import theano.tensor as T

# atari learning environment imports
from ale_python_interface import ALEInterface

# custom imports
import file_utils
import screen_utils
import feature_extractors
import learning_agents
from mlp import MLP, HiddenLayer, OutputLayer

MAX_FEATURES = 8

def train(gamepath, 
          n_episodes=10000, 
          display_screen=False, 
          record_weights=True, 
          reduce_exploration_prob_amount=True,
          n_frames_to_skip=4,
          exploration_prob=.5,
          verbose=True,
          discount=.99,
          learning_rate=.01):
    """
    :description: trains an agent to play a game 

    :type gamepath: string 
    :param gamepath: path to the binary of the game to be played

    :type agent: subclass RLAlgorithm
    :param agent: the algorithm/agent that learns to play the game

    :type n_episodes: int 
    :param n_episodes: number of episodes of the game on which to train
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
    real_actions = ale.getMinimalActionSet()
    actions = np.arange(len(real_actions))

    features = T.dvector('features')
    action = T.lscalar('action')
    reward = T.dscalar('reward')
    next_features = T.dvector('next_features')

    hidden_layer_1 = HiddenLayer(n_vis=MAX_FEATURES, n_hid=MAX_FEATURES / 2, layer_name='hidden1', activation='relu')
    hidden_layer_2 = HiddenLayer(n_vis=MAX_FEATURES / 2, n_hid=len(actions), layer_name='hidden2', activation='relu')
    # output_layer = OutputLayer(layer_name='output1', activation='relu')
    layers = [hidden_layer_1, hidden_layer_2]
    mlp = MLP(layers, discount=discount, learning_rate=learning_rate)
    loss, updates = mlp.get_loss_and_updates(features, action, reward, next_features)

    train_model = theano.function(
                    [theano.Param(features, default=np.zeros(MAX_FEATURES)),
                    theano.Param(action, default=0),
                    theano.Param(reward, default=0),
                    theano.Param(next_features, default=np.zeros(MAX_FEATURES))],
                    outputs=loss,
                    updates=updates,
                    mode='FAST_RUN')
    rewards = []
    losses = []
    best_reward = 0
    preprocessor = screen_utils.RGBScreenPreprocessor()
    feature_extractor = feature_extractors.NNetOpenCVBoundingBoxExtractor(MAX_FEATURES)
    for episode in xrange(n_episodes):

        total_reward = 0
        action = 1
        counter = 0
        reward = 0
        loss = 0
        previous_param_0 = None
        lives = ale.lives()
        screen = np.zeros((preprocessor.dim, preprocessor.dim, preprocessor.channels))
        state = { "screen" : screen, "objects" : None, "prev_objects": None, "features": np.zeros(MAX_FEATURES)}
        
        while not ale.game_over():
            if counter % n_frames_to_skip != 0:
                counter += 1
                reward += ale.act(real_actions[action])
                continue

            counter += 1

            features = state["features"]
            if random.random() < exploration_prob: 
                action = random.choice(actions)
            else:
                action = T.argmax(mlp.fprop(features)).eval()
            
            reward += ale.act(real_actions[action])
            if ale.lives() < lives: 
                lives = ale.lives()
                reward -= 1

            next_screen = ale.getScreenRGB()
            next_screen = preprocessor.preprocess(next_screen)
            next_state = {"screen": next_screen, "objects": None, "prev_objects": state["objects"]}
            next_features = feature_extractor(next_state)
            loss += train_model(features, action, reward, next_features)
            next_state["features"] = next_features
            state = next_state
            
            if verbose and counter % 53 == 0:
                print('*' * 15 + ' training information ' + '*' * 15) 
                print('episode: {}'.format(episode))
                print('reward: \t{}'.format(reward))
                print('action: \t{}'.format(real_actions[action]))
                param_info = [(p.eval(), p.name) for p in mlp.get_params()]
                for index, (val, name) in enumerate(param_info):
                    if previous_param_0 is None and index == 0:
                        previous_param_0 = val
                    print('parameter {} value: \n{}'.format(name, val))
                    if index == 0:
                        diff = val - previous_param_0
                        print('difference from previous param {}: \n{}'.format(name, diff))
                print('features: \t{}'.format(features))
                print('next_features: \t{}'.format(next_features))
                print('*' * 52)
                print('\n')

            total_reward += reward
            reward = 0
        losses.append(loss)
        rewards.append(total_reward)
        if total_reward > best_reward and record_weights:
            best_reward = total_reward
            file_utils.save_model(mlp, 'weights/mlp_2.pkl')
            print("best reward!: {}".format(total_reward))

        if episode != 0 and episode % 25 == 0 and record_weights:
            file_utils.save_rewards(rewards)
            file_utils.save_model(mlp.layers[0], 'weights/hidden.pkl')

        if exploration_prob > .1:
            exploration_prob -= reduce_exploration_prob_amount
        
        print('episode: {} ended with score: {}\tloss: {}'.format(episode, rewards[-1], losses[-1]))
        ale.reset_game()
    return rewards

if __name__ == '__main__':

    base_dir = 'roms'
    game = 'breakout.bin'
    gamepath = os.path.join(base_dir, game)
    rewards = train(gamepath, 
                    n_episodes=10000, 
                    display_screen=False, 
                    record_weights=True, 
                    reduce_exploration_prob_amount=.0004,
                    n_frames_to_skip=4,
                    exploration_prob=.5,
                    verbose=True,
                    discount=.99,
                    learning_rate=.05)
