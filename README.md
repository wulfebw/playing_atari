![alt text](https://raw.githubusercontent.com/wulfebw/playing_atari/master/media/breakout.gif "Playing Breakout")  

scripts/build_agent.py: Constructs our different learning agents (q-learning, q-learning w/ replay memory, sarsa, sarsa-lambda)  with various parameters

scripts/feature_extractors.py: Contains all of our different feature extractors (blob/OpenCV)

scripts/learning_agents.py: Contains reinforcement learning code and logic

scripts/mlp.py: implementation of deep learning network using Theano

scripts/eligibility_traces.py: Implements eligibility trace for sarsa-lambda agent

scripts/replay_memory.py: Contains replay memory code and logic

scripts/file_utils.py and scripts/screen_utils.py: Contains various screen processing and file loading/saving methods invoked by our code

These files were other utilities that we used for our analysis:

scripts/weight_util.py: used to print weight values for a parameterized feature template. Contains documentation for how to run/use

scripts/random_baselines.py: used to generate random baseline scores

tests/test_linear_mdps.py: used to test our various agents on simpler MDPs

tests/plot_rewards.py: used to plot rewards resulting from the above

tests/test-hm.py: used to generate Q-value heat maps and best action maps
