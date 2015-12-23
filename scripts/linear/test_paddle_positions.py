
import scripts.common.feature_extractors as feature_extractors
import scripts.common.file_utils as file_utils
import matplotlib.pyplot as plt

def test_paddle_positions():

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

if __name__ == '__main__':
    test_paddle_positions()
