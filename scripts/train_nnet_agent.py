"""
:description: train an agent to play a game

:development: 
    :problems:
    1. this does not work
    2. the gradients are either too small or do not exist at all
    3. i have no confidence in the feature extractor 
    4. i have no confidence in the network implementation
    5. i have no confidence in the training function
    
    :solutions:
    1. find a way to test 3,4,5 and write tests
        3. write tests for the feature extractor that emphasize edge cases
            - ball and paddle touching etc 
        4. write tests for network
            - go through the math for a small network example
            - write a test for that example
            - need examples testing 
                - both output possibilities
                - 0-valued features
                - negative-valued features
                - positive-valued features
                - negative-valued reward
                - positive-valued reward
                - multiple, consective updates
        5. not sure on this one
    2. additional questions
        1. shared variables?
        2. disconnecting gradients?
        3. why is the output layer import in preventing the argmax error?
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
MOVE_RIGHT_ACTION_VALUE = 3
MOVE_LEFT_ACTION_VALUE = 4

def train(gamepath, 
          n_episodes=10000, 
          display_screen=False, 
          record_weights=True, 
          reduce_exploration_prob_amount=True,
          n_frames_to_skip=4,
          exploration_prob=.3,
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
    actions = np.array([0,1]) # ale.getMinimalActionSet()

    features = T.dvector('features')
    action = T.lscalar('action')
    reward = T.dscalar('reward')
    next_features = T.dvector('next_features')

    hidden_layer = HiddenLayer(n_vis=MAX_FEATURES, n_hid=len(actions), layer_name='hidden', activation='relu')
    #output_layer = OutputLayer(layer_name='output1', activation='relu')
    layers = [hidden_layer]
    mlp = MLP(layers, discount=discount, learning_rate=learning_rate)
    loss, updates = mlp.get_loss_and_updates(features, action, reward, next_features)

    train_model = theano.function(
                    [theano.Param(features, default=np.zeros(MAX_FEATURES)),
                    theano.Param(action, default=0),
                    theano.Param(reward, default=0),
                    theano.Param(next_features, default=np.zeros(MAX_FEATURES))],
                    [loss],
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
        lives = ale.lives()
        screen = np.zeros((preprocessor.dim, preprocessor.dim, preprocessor.channels))
        state = { "screen" : screen, "objects" : None, "prev_objects": None, "features": np.zeros(MAX_FEATURES)}
        
        while not ale.game_over():
            counter += 1
            if counter % n_frames_to_skip != 0:
                reward += ale.act(action)
                continue

            features = state["features"]
            if random.random() < exploration_prob: 
                action = random.choice(actions)
            else:
                action = T.argmax(mlp.fprop(features)).eval()
            if action == 0: action = MOVE_RIGHT_ACTION_VALUE
            if action == 1: action = MOVE_LEFT_ACTION_VALUE
            real_action = action
            reward += ale.act(action)
            if action == MOVE_RIGHT_ACTION_VALUE: action = 0
            if action == MOVE_LEFT_ACTION_VALUE: action = 1 

            if ale.lives() < lives: 
                lives = ale.lives()
                reward -= 1

            next_screen = ale.getScreenRGB()
            next_screen = preprocessor.preprocess(next_screen)
            next_state = {"screen": next_screen, "objects": None, "prev_objects": state["objects"]}
            next_features = feature_extractor(next_state)
            loss += train_model(features, action, reward, next_features)[0]
            next_state["features"] = next_features
            state = next_state
            
            if verbose and counter % 500 == 0:
                print('\nreward: {}'.format(reward))
                print('action: {}'.format(real_action))
                param_info = [(p.eval(), p.name) for p in mlp.get_params()]
                for (val, name) in param_info:
                    print('parameter {} value: {}'.format(name, val))
                print('features: {}'.format(features))
                print('next_features: {}\n'.format(next_features))

            total_reward += reward
            reward = 0
        losses.append(loss)
        rewards.append(total_reward)
        if total_reward > best_reward and record_weights:
            best_reward = total_reward
            file_utils.save_model(mlp, 'weights/mlp_2.pkl')
            print("best reward!: {}".format(total_reward))

        if episode != 0 and episode % 50 == 0 and record_weights:
            file_utils.save_rewards(rewards)
            file_utils.save_model(mlp, 'weights/mlp_2.pkl')

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
                    learning_rate=.01)
