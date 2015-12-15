
import feature_extractors
import file_utils
import matplotlib.pyplot as plt

ball_pos = ((67,17),(72,22))
prev_ball_pos = ((68,18),(73,23))
weights = file_utils.load_weights()
mbb = feature_extractors.MockBoundingBoxExtractor(ball_pos,prev_ball_pos)
domain = []
range_weights = []
actions = [3,4]
for x in range(0, 100):
	
	state = {}
	best_score = None
	features = mbb.get_features_paddle_x(state, actions, x)	
	for feature_set in features:
		score = 0
        	for f, v in feature_set:
            		score += weights[f] * v
        		if best_score == None or score > best_score:
				best_score = score
	if best_score != None:
		domain.append(x)
		range_weights.append(best_score)
plt.scatter(domain, range_weights)
plt.show()