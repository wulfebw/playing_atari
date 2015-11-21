"""
:description: train an agent to play a game

:best weights: 2015-11-15T16:41:51:686494.pkl
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
			feature_extractor=feature_extractors.OpenCVBoundingBoxExtractor,
			discount=0.99,
			explorationProb=.0,
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
	ale = ALEInterface()
	ale.loadROM(gamepath)
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

def train_agent(gamepath, agent, n_episodes=1000, display_screen=True):
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
	best_reward = -5
	# train the agent
	for episode in xrange(n_episodes):
		total_reward = 0

		# need a preprocessor with some state
		preprocessor = screen_utils.RGBScreenPreprocessor()

		# let's just say the start screen is all zeros and our first action is 0
		screen = np.zeros((preprocessor.dim, preprocessor.dim, preprocessor.channels))
		state = { "screen" : screen, "objects" : None, "prev_objects": None, "prev_action": 0 }
		action = 1
		counter = 0
		reward = 0
		lives = ale.lives()
		# each episode consists of a game
		while not ale.game_over():
			counter += 1
			if counter % 4 != 0:
				reward += ale.act(action)
				continue

			# 1. retrieve the screen for the current frame, this amounts to the state
			new_screen = ale.getScreenRGB()

			# 2. preprocess the screen
			new_preprocessed_screen = preprocessor.preprocess(new_screen)
	
			# 3. request an action from the agent
			prev_objects = state["objects"]
			new_state = { "screen": new_preprocessed_screen, "objects": None , 
						"prev_objects": state["objects"], "prev_action": state["prev_action"]} 
			action = agent.getAction(new_state)
			new_state["prev_action"] = action

			# 4. perform that action and receive the corresponding reward
			reward += ale.act(action)
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
			#agent.incorporateFeedback(state, action, reward, new_state)

			# 6. set the new screen to be the old screen
			state = new_state

			reward = 0

		# 7. if we had a new record, save the feature weights
		if total_reward > best_reward:
			best_reward = total_reward
			file_utils.save_weights(agent.weights)
			print("best reward!: {}".format(total_reward))

		if episode != 0 and episode % 1000 == 0:
			file_utils.save_weights(agent.weights, episodic=True)

		if agent.explorationProb > .1:
			agent.explorationProb -= .02
		rewards.append(total_reward)
		print('episode: {} ended with score: {}'.format(episode, total_reward))
		ale.reset_game()
	return rewards

def main_theano_sign_lang_var_len():
	"""
	:description: this trains a model on the sign language data as well, but accounts for variable length sequences and processes batches.
	"""
	print('loading data...')
	n_input_at_each_timestep = 10
	n_classes = 97	# no base 0 considered, there are just 98 of them. May need to be 97
	
	X, y = sign_lang.load_data_from_aggregate_file()
	X, masks = sign_lang.pad_data_to_max_sample_length(X)
	X = X.astype(theano.config.floatX)
	masks = masks.astype(theano.config.floatX)
	X = np.swapaxes(X, 0, 1)
	masks = np.swapaxes(masks, 0, 1)

	split_idx = int(.8 * X.shape[1])

	X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
	masks = theano.shared(np.asarray(masks, dtype=theano.config.floatX), borrow=True)
	y = theano.shared(y, borrow=True)

	trainset_masks = masks[:, :split_idx, :]
	testset_masks = masks[:, split_idx:, :]
	
	trainset_X, trainset_y = X[:, :split_idx, :], y[:split_idx]
	testset_X, testset_y = X[:, split_idx:, :], y[split_idx:]

	index = T.lscalar()
	x = T.tensor3('x')
	target = T.lvector('target')
	print_x = theano.printing.Print('\nx')(x)
	print_target = theano.printing.Print('target')(target)
	mask = T.tensor3('mask')

	print('building model...')

	lstm_1_filepath = '/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/models/lstm_1.save'
	lstm_2_filepath = '/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/models/lstm_2.save'
	lstm_3_filepath = '/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/models/lstm_3.save'
	softmax_filepath = '/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/models/softmax_1.save'
	
	lstm_1 = load_model(lstm_1_filepath)
	# lstm_2 = load_model(lstm_2_filepath)
	# lstm_3 = load_model(lstm_3_filepath)
	softmax = load_model(softmax_filepath)


	#lstm_1 = variable_length_sequence_lstm.LSTM(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='rec_1', return_indices=[-1], dropout_prob=0.3)
	#lstm_2 = variable_length_sequence_lstm.LSTM(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='rec_2', return_indices=None, dropout_prob=0.3)
	#lstm_3 = variable_length_sequence_lstm.LSTM(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='rec_3', return_indices=[-1], dropout_prob=0.3)
	#softmax = variable_length_sequence_lstm.Softmax(n_vis=n_input_at_each_timestep, n_classes=n_classes)

	# layers = [lstm_1, lstm_2, lstm_3, softmax]
	layers = [lstm_1, softmax]

	cost_expr = variable_length_sequence_lstm.Softmax.negative_log_likelihood
	rnn = variable_length_sequence_lstm.MLP(layers, cost=cost_expr, return_indices=[-1])

	cost, updates = rnn.get_cost_updates(x, target, mask, learning_rate=0.0005)

	batch_size = 35

	print('building trainer...')
	trainer = theano.function(
		[index],
		[cost],
		updates=updates,
		givens={
			x: trainset_X[:, index * batch_size: (index + 1) * batch_size],
			target: trainset_y[index * batch_size: (index + 1) * batch_size],
			mask: trainset_masks[:, index * batch_size: (index + 1) * batch_size]
		},
		mode='FAST_RUN'
	)

	errors = rnn.layers[-1].errors(target)
	validate_model = theano.function(
		inputs=[index],
		outputs=[cost, errors],
		givens={
			x: testset_X[:, index * batch_size: (index + 1) * batch_size],
			target: testset_y[index * batch_size: (index + 1) * batch_size],
			mask: testset_masks[:, index * batch_size: (index + 1) * batch_size]
		},
		mode='FAST_RUN'
	)

	print('training model...')
	n_train_examples = trainset_X.shape.eval()[1]
	n_test_examples = testset_X.shape.eval()[1]

	n_epochs = 1000
	lowest_cost = -1
	n_train_batches = int(trainset_X.shape.eval()[1] / float(batch_size))
	n_validation_batches = int(testset_X.shape.eval()[1] / float(batch_size))
	for epoch in range(n_epochs):
		costs = []
		#random_indices = get_random_indices(max_index=n_train_examples - 1, samples_per_epoch=100)

		for sample_idx in range(n_train_batches):
		# for sample_idx in random_indices:
			costs.append(trainer(sample_idx)[0])
		avg_cost = np.mean(costs)
		print('training cost for epoch {0}: {1}'.format(epoch, avg_cost))

		if lowest_cost == -1 or avg_cost < lowest_cost * 0.99:
			lowest_cost = avg_cost
			run_validation = True
			save_model(lstm_1, lstm_1_filepath)
			# save_model(lstm_2, lstm_2_filepath)
			# save_model(lstm_3, lstm_3_filepath)
			save_model(softmax, softmax_filepath)

		predictions = []
		if run_validation:
			print('\nvalidation')
			for sample_idx in range(n_validation_batches):
				predictions.append(validate_model(sample_idx)[1])
			accuracy = (1 - np.mean(predictions)) * 100
		 	print('accuracy for epoch {0}: {1}%'.format(epoch, accuracy))
		 	run_validation = False

	# print('finished training, final stats:\nfinal cost: {0}\naccuracy: {1}%'.format(np.mean(costs), accuracy))
	print('finished training, final stats:\nfinal cost: {0}'.format(np.mean(costs)))

	for layer in rnn.layers:
		for param in layer.params:
			print('{}: {}'.format(param.name, param.get_value()))

if __name__ == '__main__':

	base_dir = 'roms'
	game = 'breakout.bin'
	gamepath = os.path.join(base_dir, game)
	agent = get_agent(gamepath)
	rewards = train_agent(gamepath, agent)
