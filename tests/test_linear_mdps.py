import sys, os, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))
import learning_agents

step_min = 0.001
step_dec = 0.0000005
explore_dec = 0.00001
explore_min = 0.05

# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(mdp, rl, numTrials=10, maxIterations=1000, verbose=False,
             sort=False, rFile=None):
    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

    totalR = 0.0
    numR = 0.0
    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
	numR += 1
        state = mdp.startState()
        sequence = [state]
        totalDiscount = 1
        totalReward = 0
        for _ in range(maxIterations):
            action = rl.getAction(state)
            transitions = mdp.succAndProbReward(state, action)
            if sort: transitions = sorted(transitions)
            if len(transitions) == 0 or mdp.isEnd(state):
                rl.incorporateFeedback(state, action, 0, None)
                break

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

            rl.incorporateFeedback(state, action, reward, newState)
            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount()
            state = newState
	totalRewards.append(totalReward)
        totalR += totalReward
	if verbose and trial % 100 == 0:
            print "Trial %d (totalReward = %s)" % (trial, totalReward)
	    print "Average: %2.4f" % (totalR/numR)
	    print "Sliding (100): %2.4f" % (sum(totalRewards[-100:])/(len(totalRewards[-100:]) + 1.0))
	    print "Step: %2.5f; Explore: %2.4f" % (rl.stepSize, rl.explorationProb)
	    if rFile != None:
		rFile.write("" + str(sum(totalRewards[-100:])/(len(totalRewards[-100:]) + 1.0)))
		rFile.write("\n")
	rl.explorationProb -= explore_dec
	if (rl.explorationProb < explore_min):
		rl.explorationProb = explore_min
	rl.stepSize -= step_dec
	if (rl.stepSize < step_min):
		rl.stepSize = step_min
    return totalRewards

# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action): raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)

class GridSearchMDP(MDP):
    def __init__(self, n=10, hasMagic=False):
        self.n = n
	self.magic = hasMagic
	if self.magic:
		self.magicSquare = (3,4)
    def startState(self): return (random.randint(2, self.n - 4) + 0.0, random.randint(2, self.n - 4) + 0.0, False)
    def isEnd(self, state): return state[0] == 0 or state[0] >= self.n or state[1] == 0 or state[1] >= self.n
    def actions(self, state): return [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1)]
    def succAndProbReward(self, state, action):
        newStateR = (state[0] + action[0], state[1] + action[1])
        nextReward = 0
	visited = state[2]
	if newStateR[0] == self.magicSquare[0] and newStateR[1] == self.magicSquare[1]:
		visited = True
	newState = (newStateR[0], newStateR[1], visited)
        if self.isEnd(newState):
                if newState[0] == 0 or newState[1] == 0 or (newState[0] == self.n and newState[1] != self.n):
                        nextReward = -1
                elif newState[0] == self.n and newState[1] == self.n:
                        nextReward = 5
                elif newState[1] == self.n:
                        nextReward = 1
		if newState[2]:
			nextReward += 10
        return  [(newState, 1.0, nextReward)]
    def discount(self): return 0.99

def MdpFExtractor(state, action):
	features = [ (((state[0],state[1]),action), 1) ]
	return features

###  parameters common to all agents ###
DISCOUNT = .99
EXPLORATION_PROBABILITY = 0.35
STEP_SIZE = .02
MAX_GRADIENT = 10
########################################
######## sarsa lambda parameters #######
THRESHOLD = .01
DECAY = .9
########################################
def build_sarsa_lambda_agent(actions):
    print 'building sarsa lambda agent...'
    return learning_agents.SARSALambdaLearningAlgorithm(
                actions=actions,
                discount=DISCOUNT,
                featureExtractor=MdpFExtractor,
                explorationProb=EXPLORATION_PROBABILITY,
                stepSize=STEP_SIZE,
		threshold=THRESHOLD,
		decay=DECAY,
                maxGradient=MAX_GRADIENT)

## Qlearning replay memory parameters ##
REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_SAMPLE_SIZE = 3
NUM_STATIC_TARGET_UPDATE_STEPS = 2500
NUM_CONSECUTIVE_RANDOM_ACTIONS = 0 # 0 denotes only taking a random action once
########################################
def build_q_learning_replay_memory_agent(actions):
    print 'building Q-learning agent with replay memory...'
    featureExtractor = MdpFExtractor
    return learning_agents.QLearningReplayMemoryAlgorithm(
                actions=actions,
                discount=DISCOUNT,
                featureExtractor=featureExtractor,
                explorationProb=EXPLORATION_PROBABILITY,
                stepSize=STEP_SIZE,
                replay_memory_size=REPLAY_MEMORY_SIZE,
                replay_memory_sample_size=REPLAY_MEMORY_SAMPLE_SIZE,
                num_static_target_update_steps=NUM_STATIC_TARGET_UPDATE_STEPS,
                maxGradient=MAX_GRADIENT,
                num_consecutive_random_actions=NUM_CONSECUTIVE_RANDOM_ACTIONS)
########################################

def main():
	mdp = GridSearchMDP(hasMagic = True)
	actions = mdp.actions(mdp.startState())
	#agent = build_sarsa_lambda_agent(actions)
	agent = build_q_learning_replay_memory_agent(actions)
	with open("replay.out", "w") as rfile:
		simulate(mdp, agent, numTrials=100000, verbose=True, rFile=rfile)

main()
