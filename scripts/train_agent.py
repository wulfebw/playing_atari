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

    :type gamepath: string 
    :param gamepath: path to the binary of the game to be played

    :type ReinforcementLearner: subclass of learning_agents.RLAlgorithm
    :param ReinforcementLearner: the algorithm/agent that will learn to play the game

    :type discount: float
    :param discount: discount factor applied to rewards over time

    :type explorationProb: float
    :param explorationProb: the probability that the agent takes a random action. 
                            just assuming epsilon-greedy.

    :type load_weights: boolean
    :param load_weights: whether or not to load in stored weights for the agent

    """
    
    # 3 goes right
    # 4 goes left
    legal_actions = np.array([1,3,4]) # ale.getLegalActionSet()

    # 2. instantiate and return agent
    fe = feature_extractor()
    agent = learning_algorithm(actions=legal_actions,
                                featureExtractor=fe,
                                discount=discount,
                                explorationProb=explorationProb,
                                stepSize=stepSize,
                                maxGradient=maxGradient)

    # 3. load and set weights if we have them
    if load_weights:
        weights = file_utils.load_weights()
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

    rewards = [-5]
    best_reward = 5
    for episode in xrange(n_episodes):

        total_reward = 0
        action = 1
        counter = 0
        reward = 0
        lives = ale.lives()
        preprocessor = screen_utils.RGBScreenPreprocessor()
        screen = np.zeros((preprocessor.dim, preprocessor.dim, preprocessor.channels))
        state = { "screen" : screen, "objects" : None, "prev_objects": None, "prev_action": 0, "action": 0 }
        
        while not ale.game_over():
            counter += 1
            if counter % n_frames_to_skip != 0:
                reward += ale.act(action)
                continue

            action = agent.getAction(state)
            reward += ale.act(action)
            # if ale.lives() < lives:
            #   lives = ale.lives()
            #   reward -= 1
            total_reward += reward

            new_screen = ale.getScreenRGB()
            new_preprocessed_screen = preprocessor.preprocess(new_screen)
            new_state = {"screen": new_preprocessed_screen, "objects": None, "prev_objects": state["objects"], "prev_action": state["action"], "action": action}
            agent.incorporateFeedback(state, action, reward, new_state)
            state = new_state
            reward = 0

        rewards.append(total_reward)
        if total_reward > best_reward and record_weights:
            best_reward = total_reward
            file_utils.save_weights(agent.weights)
            print("best reward!: {}".format(total_reward))

        if episode != 0 and episode % 1000 == 0 and record_weights:
            file_utils.save_rewards(rewards)
            file_utils.save_weights(agent.weights, episodic=True)

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
                    discount=0.999,
                    explorationProb=0.0,
                    stepSize=0.005,
                    maxGradient=1)
    rewards = train_agent(gamepath, 
                        agent, 
                        n_episodes=20000, 
                        display_screen=True, 
                        record_weights=False, 
                        reduce_exploration_prob_amount=.0001,
                        n_frames_to_skip=4)
