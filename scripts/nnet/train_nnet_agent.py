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
import scripts.common.file_utils as file_utils
import scripts.common.screen_utils as screen_utils
import scripts.common.feature_extractors as feature_extractors
from scripts.common.replay_memory import ReplayMemory
from qnetwork import QNetwork, HiddenLayer, OutputLayer

######## training parameters #########
NUM_EPISODES = 10000
NNET_INPUT_DIMENSION = 8
EXPLORATION_REDUCTION_AMOUNT = .0005
MINIMUM_EXPLORATION_EPSILON = .05
NUM_FRAMES_TO_SKIP = 4
EXPLORATION_PROBABILITY = 1
DISCOUNT = .99
LEARNING_RATE = 0.004
FROZEN_TARGET_UPDATE_PERIOD = 5
USE_REPLAY_MEMORY = True
######################################

########## training options ##########
LOAD_WEIGHTS = False
DISPLAY_SCREEN = False
PRINT_TRAINING_INFO_PERIOD = 10
NUM_EPISODES_AVERAGE_REWARD_OVER = 100
RECORD_WEIGHTS = False
RECORD_WEIGHTS_PERIOD = 25
VERBOSE = True
######################################


def train(gamepath, n_episodes,  display_screen,  record_weights,  reduce_exploration_prob_amount, n_frames_to_skip, exploration_prob, verbose, discount, learning_rate, load_weights, frozen_target_update_period, use_replay_mem):
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
    ale.setInt("frame_skip", n_frames_to_skip)
    # real actions for breakout are [0,1,3,4]
    real_actions = ale.getMinimalActionSet()

    # use a list of actions [0,1,2,3] to index into the array of real actions
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
        hidden_layer_1 = HiddenLayer(n_vis=NNET_INPUT_DIMENSION, 
            n_hid=NNET_INPUT_DIMENSION, layer_name='hidden1', activation='relu')
        hidden_layer_2 = HiddenLayer(n_vis=NNET_INPUT_DIMENSION, 
            n_hid=NNET_INPUT_DIMENSION, layer_name='hidden2', activation='relu')
    hidden_layer_3 = HiddenLayer(n_vis=NNET_INPUT_DIMENSION, 
            n_hid=len(actions), layer_name='hidden3', activation='relu') 
    # the output layer is currently necessary when using tanh units in the
    # hidden layer in order to prevent a theano warning
    # currently the relu unit setting of the hidden and output layers is leaky w/ alpha=0.01
    output_layer = OutputLayer(layer_name='output', activation='relu')

    # pass a list of layers to the constructor of the network (here called "mlp")
    layers = [hidden_layer_1, hidden_layer_2, hidden_layer_3, output_layer]
    qnetwork = QNetwork(layers, discount=discount, learning_rate=learning_rate)

    # this call gets the symbolic output of the network
    # along with the parameter updates expected
    loss, updates = qnetwork.get_loss_and_updates(features, action, reward, next_features)

    # this defines the theano symbolic function used to train the network
    # 1st argument is a list of inputs, here the symbolic variables above
    # 2nd argument is the symbolic output expected
    # 3rd argument is the dictionary of parameter updates
    # 4th argument is the compilation mode
    train_model = theano.function(
                    [theano.Param(features, default=np.zeros(NNET_INPUT_DIMENSION)),
                    theano.Param(action, default=0),
                    theano.Param(reward, default=0),
                    theano.Param(next_features, default=np.zeros(NNET_INPUT_DIMENSION))],
                    outputs=loss,
                    updates=updates,
                    mode='FAST_RUN')

    sym_action = qnetwork.get_action(features)
    get_action = theano.function([features], sym_action)

    # some containers for collecting information about the training processes 
    rewards = []
    losses = []
    best_reward = 4
    sequence_examples = []
    sampled_examples = []

    # the preprocessor and feature extractor to use
    preprocessor = screen_utils.RGBScreenPreprocessor()
    feature_extractor = feature_extractors.NNetOpenCVBoundingBoxExtractor(max_features=MAX_FEATURES)

    if use_replay_mem:
        replay_mem = ReplayMemory()
    # main training loop, each episode is a full playthrough of the game
    for episode in xrange(n_episodes):

        # this implements the frozen target component of the network
        # by setting the frozen layers of the network to a copy of the current layers
        if episode % frozen_target_update_period == 0:
            qnetwork.frozen_layers = copy.deepcopy(qnetwork.layers)


        # some variables for collecting information about this particular run of the game
        total_reward = 0
        action = 1
        counter = 0
        reward = 0
        loss = 0
        previous_param_0 = None

        # lives here is used for the reward heuristic of subtracting 1 from the reward 
        # when we lose a life. currently commented out this functionality because
        # i think it might not be helpful.
        lives = ale.lives()

        # the initial state of the screen and state
        screen = np.zeros((preprocessor.dim, preprocessor.dim, preprocessor.channels))
        state = { "screen" : screen, "objects" : None, "prev_objects": None, "features": np.zeros(MAX_FEATURES)}
        
        # start the actual play through of the game
        while not ale.game_over():
            counter += 1

            # get the current features, which is the representation of the state provided to 
            # the "agent" (here just the network directly)
            features = state["features"]

            # epsilon greedy action selection (note that exploration_prob is reduced by
            # reduce_exploration_prob_amount after every game)
            if random.random() < exploration_prob: 
                action = random.choice(actions)
            else:
                # to choose an action from the network, we fprop 
                # the current state and take the argmax of the output
                # layer (i.e., the action that corresponds to the 
                # maximum q value)
                action = get_action(features)

            # take the action and receive the reward
            reward += ale.act(real_actions[action])

            # this is commented out because i think it might not be helpful
            if ale.lives() < lives: 
                 lives = ale.lives()
                 reward -= 1


            # get the next screen, preprocess it, initialize the next state
            next_screen = ale.getScreenRGB()
            next_screen = preprocessor.preprocess(next_screen)
            next_state = {"screen": next_screen, "objects": None, "prev_objects": state["objects"]}

            # get the features for the next state
            next_features = feature_extractor(next_state, action=None)

            if use_replay_mem:
                sars_tuple = (features, action, reward, next_features)
                replay_mem.store(sars_tuple)
                num_samples = 5 if replay_mem.isFull() else 1
                for i in range(0, num_samples):
                    random_train_tuple = replay_mem.sample()
                    loss += train_model(*random_train_tuple)

                # collect for pca
                sequence_examples.append(list(sars_tuple[0]) + [sars_tuple[1]] \
                         + [sars_tuple[2]] + sars_tuple[3])
                sequence_examples = sequence_examples[-100:]
                sampled_examples.append(list(random_train_tuple[0]) + [random_train_tuple[1]] \
                        + [random_train_tuple[2]] + random_train_tuple[3])
                sampled_examples = sampled_examples[-100:]
            else:
                # call the train model function
                loss += train_model(features, action, reward, next_features)
            # prepare for the next loop through the game
            next_state["features"] = next_features
            state = next_state
                
            # weird counter value to avoid interaction with any other counter
            # loop that might be added, not necessary right now
            if verbose and counter % PRINT_TRAINING_INFO_PERIOD == 0:
                print('*' * 15 + ' training information ' + '*' * 15) 
                print('episode: {}'.format(episode))
                print('reward: \t{}'.format(reward))
                print('avg reward: \t{}'.format(np.mean(rewards)))
                print 'avg reward (last 25): \t{}'.format(np.mean(rewards[-NUM_EPISODES_AVERAGE_REWARD_OVER:]))
                print('action: \t{}'.format(real_actions[action]))
                print('exploration prob: {}'.format(exploration_prob))
                
                param_info = [(p.eval(), p.name) for p in qnetwork.get_params()]
                for index, (val, name) in enumerate(param_info):
                    if previous_param_0 is None and index == 0:
                        previous_param_0 = val
                    print('parameter {} value: \n{}'.format(name, val))
                    if index == 0:
                        diff = val - previous_param_0
                        print('difference from previous param {}: \n{}'.format(name, diff))

                print('features: \t{}'.format(features))
                print('next_features: \t{}'.format(next_features))

                scaled_sequence = preprocessing.scale(np.array(sequence_examples))
                scaled_sampled = preprocessing.scale(np.array(sampled_examples))
                pca = PCA()
                _ = pca.fit_transform(scaled_sequence)
                print('variance explained by first component for sequence: {}%'.format(pca. \
                    explained_variance_ratio_[0] * 100))
                _ = pca.fit_transform(scaled_sampled)
                print('variance explained by first component for sampled: {}%'.format(pca. \
                    explained_variance_ratio_[0] * 100))

                print('*' * 52)
                print('\n')

            # collect info and total reward and also reset the reward to 0 if we reach this point
            total_reward += reward
            reward = 0
        # collect stats from this game run    
        losses.append(loss)
        rewards.append(total_reward)
    
        # if we got a best reward, inform the user 
        if total_reward > best_reward:
            best_reward = total_reward
            print("best reward!: {}".format(total_reward))

        # record the weights if record_weights=True
        # must record the weights of the indiviual layers
        # only save hidden layers b/c output layer does not have weights
        if episode != 0 and episode % RECORD_WEIGHTS_PERIOD == 0 and record_weights:
            file_utils.save_rewards(rewards)
            file_utils.save_model(qnetwork.layers[0], 'weights/hidden0_{}.pkl'.format(episode))
            file_utils.save_model(qnetwork.layers[1], 'weights/hidden1_{}.pkl'.format(episode))

        # reduce exploration policy over time
        if exploration_prob > MINIMUM_EXPLORATION_EPSILON:
            exploration_prob -= reduce_exploration_prob_amount
        
        # inform user of how the episode went and reset the game
        print('episode: {} ended with score: {}\tloss: {}'.format(episode, rewards[-1], losses[-1]))
        ale.reset_game()

    # return the list of rewards attained
    return rewards

if __name__ == '__main__':
    base_dir = 'roms'
    game = 'breakout.bin'
    gamepath = os.path.join(base_dir, game)
    rewards = train(gamepath, 
                    n_episodes=NUM_EPISODES, 
                    display_screen=False, 
                    record_weights=True, 
                    reduce_exploration_prob_amount=EXPLORATION_REDUCTION_AMOUNT,
                    n_frames_to_skip=NUM_FRAMES_TO_SKIP,
                    exploration_prob=EXPLORATION_PROBABILITY,
                    verbose=True,
                    discount=DISCOUNT,
                    learning_rate=LEARNING_RATE,
                    load_weights=False,
                    frozen_target_update_period=FROZEN_TARGET_UPDATE_PERIOD,
                    use_replay_mem=USE_REPLAY_MEMORY)


