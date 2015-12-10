"""
:description: training constants and building training agents
"""

import file_utils
import feature_extractors
import learning_agents

###  parameters common to all agents ###
ACTIONS = [0,1,3,4]
DISCOUNT = .99
EXPLORATION_PROBABILITY = 1
STEP_SIZE = .001
MAX_GRADIENT = 10
FEATURE_EXTRACTOR = feature_extractors.OpenCVBoundingBoxExtractor()
########################################

###### feature extractor options: ######
# OpenCVBoundingBoxExtractor()
# TrackingClassifyingContourExtractor()
########################################

########################################
def build_sarsa_agent():
    print 'building sarsa agent...'
    featureExtractor = FEATURE_EXTRACTOR
    return learning_agents.SARSALearningAlgorithm(
                actions=ACTIONS,
                discount=DISCOUNT, 
                featureExtractor=featureExtractor, 
                explorationProb=EXPLORATION_PROBABILITY, 
                stepSize=STEP_SIZE, 
                maxGradient=MAX_GRADIENT)
########################################

######## sarsa lambda parameters #######
THRESHOLD = .1
DECAY = .98
########################################
def build_sarsa_lambda_agent():
    print 'building sarsa lambda agent...'
    featureExtractor = FEATURE_EXTRACTOR
    return learning_agents.SARSALambdaLearningAlgorithm(
                actions=ACTIONS,
                discount=DISCOUNT, 
                featureExtractor=featureExtractor, 
                explorationProb=EXPLORATION_PROBABILITY, 
                stepSize=STEP_SIZE, 
                threshold=THRESHOLD, 
                decay=DECAY, 
                maxGradient=MAX_GRADIENT)
########################################

########################################
def build_q_learning_agent():
    print 'building Q-learning agent...'
    featureExtractor = FEATURE_EXTRACTOR
    return learning_agents.QLearningAlgorithm(
                actions=ACTIONS,
                discount=DISCOUNT, 
                featureExtractor=featureExtractor, 
                explorationProb=EXPLORATION_PROBABILITY, 
                stepSize=STEP_SIZE, 
                maxGradient=MAX_GRADIENT)
########################################

## Qlearning replay memory parameters ##
REPLAY_MEMORY_SIZE = 100
REPLAY_MEMORY_SAMPLE_SIZE = 10
########################################
def build_q_learning_replay_memory_agent():
    print 'building Q-learning agent with replay memory...'
    featureExtractor = FEATURE_EXTRACTOR
    return learning_agents.QLearningReplayMemoryAlgorithm(
                actions=ACTIONS,
                discount=DISCOUNT, 
                featureExtractor=featureExtractor, 
                explorationProb=EXPLORATION_PROBABILITY, 
                stepSize=STEP_SIZE,  
                maxGradient=MAX_GRADIENT,
                replay_memory_size=REPLAY_MEMORY_SIZE, 
                replay_memory_sample_size=REPLAY_MEMORY_SAMPLE_SIZE)
########################################

def load_agent_weights(agent, weights_filepath):
    pass
