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
STEP_SIZE = .01
MAX_GRADIENT = 10
########################################

###### feature extractor options: ######
# OpenCVBoundingBoxExtractor()
# TrackingClassifyingContourExtractor()
########################################

########################################
def build_sarsa_agent():
    print 'building sarsa agent...'
    featureExtractor = feature_extractors.OpenCVBoundingBoxExtractor()
    return learning_agents.SARSALearningAlgorithm(
                actions=ACTIONS,
                discount=DISCOUNT, 
                featureExtractor=featureExtractor, 
                explorationProb=EXPLORATION_PROBABILITY, 
                stepSize=STEP_SIZE, 
                maxGradient=MAX_GRADIENT)
########################################

######## sarsa lambda parameters #######
THRESHOLD = .01
DECAY = .5
########################################
def build_sarsa_lambda_agent():
    print 'building sarsa lambda agent...'
    featureExtractor = featureExtractor = feature_extractors.OpenCVBoundingBoxExtractor()
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
    featureExtractor = feature_extractors.TrackingClassifyingContourExtractor()
    return learning_agents.QLearningAlgorithm(
                actions=ACTIONS,
                discount=DISCOUNT, 
                featureExtractor=featureExtractor, 
                explorationProb=EXPLORATION_PROBABILITY, 
                stepSize=STEP_SIZE, 
                threshold=THRESHOLD, 
                decay=DECAY, 
                maxGradient=MAX_GRADIENT)
########################################

def load_agent_weights(agent, weights_filepath):
    pass