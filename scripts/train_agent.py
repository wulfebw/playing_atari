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
import build_agent

######## training parameters #########
NUM_EPISODES = 10000
EXPLORATION_REDUCTION_AMOUNT = .001
MINIMUM_EXPLORATION_EPSILON = .05
NUM_FRAMES_TO_SKIP = 4
######################################

########## training options ##########
DISPLAY_SCREEN = False
PRINT_TRAINING_INFO_PERIOD = 1
NUM_EPISODES_AVERAGE_REWARD_OVER = 100
RECORD_WEIGHTS = False
RECORD_WEIGHTS_PERIOD = 100
######################################


def train_agent(gamepath, agent, n_episodes, display_screen, record_weights, 
        reduce_exploration_prob_amount, n_frames_to_skip):
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
        newAction = None

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

            # if newAction is None then we are training an off-policy algorithm
            # otherwise, we are training an on policy algorithm
            if newAction is None:
                action = agent.getAction(state)
            else:
                action = newAction
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
            newAction = agent.incorporateFeedback(state, action, reward, new_state)

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
            print("Exploration probability: {}".format(agent.explorationProb))
            print('action: \t{}'.format(action))
            print('size of weights dict: {}'.format(len(agent.weights)))
            avg_feat_weight = np.mean([v for k,v in agent.weights.iteritems()])
            print('average feature weight: {}'.format(avg_feat_weight))
            
        if episode != 0 and episode % RECORD_WEIGHTS_PERIOD == 0 and record_weights:
            file_utils.save_rewards(rewards)
            file_utils.save_weights(agent.weights, filename='episode-{}-{}-weights.pkl'.format(episode, type(agent).__name__))

        if agent.explorationProb > MINIMUM_EXPLORATION_EPSILON:
            agent.explorationProb -= reduce_exploration_prob_amount
        
        print('episode: {} ended with score: {}'.format(episode, total_reward))
        ale.reset_game()
    return rewards

if __name__ == '__main__':
    game = 'breakout.bin'
    gamepath = os.path.join('roms', game)
    agent = build_agent.build_sarsa_lambda_agent()
    rewards = train_agent(gamepath, agent, 
                        n_episodes=NUM_EPISODES, 
                        display_screen=DISPLAY_SCREEN, 
                        record_weights=RECORD_WEIGHTS, 
                        reduce_exploration_prob_amount=EXPLORATION_REDUCTION_AMOUNT,
                        n_frames_to_skip=NUM_FRAMES_TO_SKIP)
