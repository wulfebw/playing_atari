"""
:description: train an agent to play a game
"""

import os
import sys

import numpy as np

# atari learning environment imports
from ale_python_interface import ALEInterface

# custom imports
import file_utils
import screen_utils
import feature_extractors
import learning_agents

PRINT_TRAINING_INFO_PERIOD = 25
RECORD_WEIGHTS_PERIOD = 100
NUM_EPISODES_AVERAGE_REWARD_OVER = 100

def get_agent(gamepath,
            learning_algorithm,
            feature_extractor,
            load_weights,
            discount,
            explorationProb,
            stepSize,
            maxGradient):
    """
    :description: instantiates an agent

    :type learning_algorithm: subclass of learning_agents.RLAlgorithm
    :param learning_algorithm: the algorithm/agent that will learn to play the game

    :type feature_extractor: a callable class returning a dictionary
    :param feature_extractor: a callable that extracts features from a state

    :type load_weights: boolean
    :param load_weights: whether or not to load in stored weights for the agent

    :type discount: float
    :param discount: discount factor applied to rewards over time

    :type explorationProb: float
    :param explorationProb: the probability that the agent takes a random action. 

    :type stepSize: float
    :param stepSize: learning rate applied to updates

    :type maxGradient: float
    :param maxGradient: maximum allowed gradient magnitude applied in gradient clipping
    """
    legal_actions = [0,1,3,4] 
    # instantiate agent
    agent = learning_algorithm(actions=legal_actions,
                                featureExtractor=feature_extractor(),
                                discount=discount,
                                explorationProb=explorationProb,
                                stepSize=stepSize,
                                maxGradient=maxGradient)

    # 3. load and set weights if we have them
    if load_weights:
        weights = file_utils.load_weights('episode-0-weights.pkl')
        if weights is not None:
            agent.weights = weights
    return agent

def train_agent(gamepath, 
                agent, 
                n_episodes=10000, 
                display_screen=True, 
                record_weights=True, 
                reduce_exploration_prob_amount=True,
                n_frames_to_skip=4):
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
    ale.setInt("frame_skip", n_frames_to_skip)

    rewards = []
    best_reward = 0
    for episode in xrange(n_episodes):
        action = 0  
        reward = 0

        total_reward = 0
        counter = 0
        lives = ale.lives()

        screen = np.zeros((32, 32, 3), dtype=np.int8)
        state = { "screen" : screen, 
                "objects" : None, 
                "prev_objects": None, 
                "prev_action": 0, 
                "action": 0 }
        
        while not ale.game_over():

            action = agent.getAction(state)
            reward += ale.act(action)
            if ale.lives() < lives:
              lives = ale.lives()
              reward -= 1
            total_reward += reward

            new_screen = ale.getScreenRGB()
            new_state = {"screen": new_screen, 
                        "objects": None, 
                        "prev_objects": state["objects"], 
                        "prev_action": state["action"], 
                        "action": action}
            agent.incorporateFeedback(state, action, reward, new_state)
            state = new_state
            reward = 0

        rewards.append(total_reward)

        if total_reward > best_reward and record_weights:
            best_reward = total_reward
            file_utils.save_weights(agent.weights)
            print("Best reward: {}".format(total_reward))

        if episode % PRINT_TRAINING_INFO_PERIOD == 0:
            print("Average reward: {}".format(np.mean(rewards)))
            print("Last 50: {}".format(np.mean(rewards[-NUM_EPISODES_AVERAGE_REWARD_OVER:])))
            print("Explore: {}".format(agent.explorationProb))
        
        if episode != 0 and episode % RECORD_WEIGHTS_PERIOD == 0 and record_weights:
            file_utils.save_rewards(rewards)
            file_utils.save_weights(agent.weights, filename='episode-{}-weights.pkl'.format(episode))

        if agent.explorationProb > .1:
            agent.explorationProb -= reduce_exploration_prob_amount
        
        print('episode: {} ended with score: {}'.format(episode, total_reward))
        ale.reset_game()
    return rewards

if __name__ == '__main__':

    base_dir = 'roms'
    game = 'breakout.bin'
    gamepath = os.path.join(base_dir, game)
    agent = get_agent(gamepath, 
                    learning_algorithm=learning_agents.QLearningAlgorithm,
                    feature_extractor=feature_extractors.OpenCVBoundingBoxExtractor,
                    load_weights=False,
                    discount=0.993,
                    explorationProb=1,
                    stepSize=0.01,
                    maxGradient=1)
    rewards = train_agent(gamepath, 
                    agent, 
                    n_episodes=10000, 
                    display_screen=True, 
                    record_weights=True, 
                    reduce_exploration_prob_amount=.001,
                    n_frames_to_skip=4)
