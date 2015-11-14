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
			learning_algorithm=learning_agents.QLearningAlgorithm,
			feature_extractor=feature_extractors.OpenCVBoundingBox,
			discount=0.999,
			explorationProb=.5,
			load_weights=True):
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
	
	# 1. load the legal actions for this game (don't see a reasonable way around this)
	# ale = ALEInterface()
	# ale.loadROM(gamepath)
	# 3 goes right
	# 4 goes left
	# 6 also goes right?
	# 7 also goes left?
	# 8 goes right also
	legal_actions = np.array([1,3,4])#ale.getLegalActionSet()

	# 2. instantiate and return agent
	fe = feature_extractor()
	agent = learning_algorithm(actions=legal_actions,
								discount=discount,
								featureExtractor=fe,
								explorationProb=explorationProb)

	# 3. load and set weights if we have them
	if load_weights:
		weights = file_utils.load_weights()
		if weights is not None:
			agent.weights = weights

	return agent

def train_agent(gamepath, agent, n_episodes=1000, display_screen=False):
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

	# load the ROM file
	ale.loadROM(gamepath)

	rewards = []
	best_reward = 0
	# train the agent
	for episode in xrange(n_episodes):
		total_reward = 0

		# need a preprocessor with some state
		preprocessor = screen_utils.RGBScreenPreprocessor()

		# let's just say the start screen is all zeros and our first action is 0
		screen = np.zeros((preprocessor.dim, preprocessor.dim, preprocessor.channels))
		state = { "screen" : screen, "objects" : None }
		action = 0
		counter = 0
		reward = 0
		lives = ale.lives()
		# each episode consists of a game
		while not ale.game_over():
			counter += 1

			# 1. retrieve the screen for the current frame, this amounts to the state
			#new_screen = ale.getScreenGrayscale()
			new_screen = ale.getScreenRGB()

			# 2. preprocess the screen
			new_preprocessed_screen = preprocessor.preprocess(new_screen)
	
			# 3. request an action from the agent

			new_state = { "screen" : new_preprocessed_screen, "objects" : None , "prev_objects" : state["objects"]} 

			action = agent.getAction(new_state)

			# 4. perform that action and receive the corresponding reward
			reward = ale.act(action)
			if ale.lives() < lives:
				lives = ale.lives()
				reward -= 1
			
			# restrict reward to {-1, 0, 1}
			if reward > 0:
				reward = 1
			elif reward < 0:
				reward = -1

			total_reward += reward

			# 5. incorporate this feedback into the agent
			agent.incorporateFeedback(state, action, reward, new_state)

			# 6. set the new screen to be the old screen
			state = new_state

			# 7. if we had a new record, save the feature weights
			if reward > best_reward:
				best_reward = reward
				file_utils.save_weights(agent.weights)
				print("best reward!: {}".format(reward))

		if agent.explorationProb > .1:
			agent.explorationProb -= .005
		rewards.append(total_reward)
		print('episode: {} ended with score: {}'.format(episode, total_reward))
		ale.reset_game()
	return rewards

if __name__ == '__main__':

	base_dir = 'roms'
	game = 'breakout.bin'
	gamepath = os.path.join(base_dir, game)
	agent = get_agent(gamepath)
	rewards = train_agent(gamepath, agent)
