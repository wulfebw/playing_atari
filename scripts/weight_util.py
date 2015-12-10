
import matplotlib.pyplot as plt
import os
import pickle
import re
import sys

def weight_key_matches(weight_key_tuple, regexes, domain_index):
	if (len(weight_key_tuple) == len(regexes)):
        	on_regex = 0
		stored_domain = None
                for regex in regexes:
                	match = regex.match(weight_key_tuple[on_regex])
                        if match == None:
				return None
			if on_regex == domain_index:
				stored_domain = float(match.group(1)) # extract for graphing
			on_regex += 1
		return stored_domain # all regexes match
	else:
		return None

def graph(weight_filename, weight_str, domain_from_regex=0, plot=True):
	domain_val = []
	range_val = []
	min = None
	max = None
	with open("weights/" + weight_filename) as weight_file:
		weights = pickle.load(weight_file)
		print weights
		key_tuple_components = weight_str.split("|")
		regexes = []
		on_regex = 0
		for regex_str in key_tuple_components:
			regexes.append(re.compile(regex_str))
			if on_regex == domain_from_regex:
				print "Adding key regex (+domain): " + regex_str
			else:
				print "Adding key regex: " + regex_str
			on_regex += 1
		for weight_key_tuple in weights:
			match_domain = weight_key_matches(weight_key_tuple, regexes, domain_from_regex)
			if match_domain != None:
				#print "Found matching tuple: " + str(weight_key_tuple)
				#print "Weight: " + str(weights[weight_key_tuple])
				#print "Domain: " + str(match_domain)
				domain_val.append(match_domain)
				range_val.append(weights[weight_key_tuple])
				roundedD = round(match_domain)
				if (max == None or roundedD > max):
					max = roundedD
				if (min == None or roundedD < min):
					min = roundedD
				if int(roundedD) == 5:
					print weights[weight_key_tuple]
	print "\n"
	domain_buckets = [None]*int(max + 1 - min)
	for i in range(0, len(domain_val)):
		d_v = domain_val[i]
		r_v = range_val[i]
		roundedD = int(round(d_v))
		if roundedD == 5:
			print r_v
		if domain_buckets[roundedD] == None:
			domain_buckets[roundedD] = [ r_v ]
		else:
			domain_buckets[roundedD].append(r_v)
	domain_val = []
	range_val = []
	print "\n"
	for i in range(0, len(domain_buckets)):
		total = 0.0
		domain_v = int(min + i)
		for range_v in domain_buckets[domain_v]:
			total += range_v
		domain_val.append(domain_v)
		if int(min + i) == 5:
			print "\n"
			print total / len(domain_buckets[domain_v])
		range_val.append(total / len(domain_buckets[domain_v]))
	if (plot):
		plt.scatter(domain_val, range_val)
		plt.xlabel("X Position (Paddle)")
		plt.ylabel("Feature Weight")
		plt.title("Paddle X-Position Feature Cross Action 3")
		plt.show()
	return domain_val, range_val

def main(argv):
	if len(argv) < 3:
		print "usage: python graph_util.y [weight_filename] [weight_regex_string] [optional_group_to_match"
		print "example: python scripts/weight_util.py 3bucket.pkl \"diff\-x\-pos\-(\-?[0-9]+)|action\-3\" 0"
		return
	if (len(argv) == 4): # override which group from which to extract domain
		graph(argv[1], argv[2], int(argv[3]))
	elif (len(argv) == 5):
		d, r = graph(argv[1], argv[2], int(argv[4]))
		d2, r2 = graph(argv[1], argv[3], int(argv[4]))
	else:
		graph(argv[1], argv[2])

main(sys.argv)
