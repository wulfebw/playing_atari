
import matplotlib.pyplot as plt

scores_r = []
with open("replay.out") as rf:
	for line in rf:
		scores_r.append(float(line))

scores_sl = []
with open("sarsa-lambda.out") as slf:
	for line in slf:
		scores_sl.append(float(line))

scores_s = []	
with open("sarsa.out") as sf:
	for line in sf:
		scores_s.append(float(line))

min_len = min([len(scores_s), len(scores_sl), len(scores_r)])
scores_s = scores_s[:min_len]
scores_sl = scores_sl[:min_len]
scores_r = scores_r[:min_len]

domain = [i for i in xrange(0, 45400, 100)]
plt.plot(domain, scores_s,label="SARSA")
plt.plot(domain, scores_sl, label="SARSA-lamda")
plt.plot(domain, scores_r, label="Q-Learning + Replay")
plt.legend(loc='lower right')
plt.xlabel("Episode number")
plt.ylabel("Average score (last 100 trials)")
plt.show()
