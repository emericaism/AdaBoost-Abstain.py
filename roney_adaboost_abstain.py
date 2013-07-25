from __future__ import division #makes everything a float
import csv
import numpy as np
import operator
import math
import random


#def reshuffle()
#def train_on_future_economists_only

def get_data(filename=str,KFold = 1):
	''' 
	Collects the data from the master file
	cleans it in the form of a dictionary of historical estimates.
	Keys are economists
	Values are List of lists containing predictions.
	'''
	global d_predictions#by economist!
	global d_weights#by date!
	global d_error#by economist!
	global d_w_correct
	global d_w_abstain
	global d_Z
	global d_alpha#by economist!
	global dates#list
	global test_dates
	global train_dates
	global d_testpredictions

	d_predictions = {}#by economist
	d_weights = {} #by date!
	#For each economist we want to know the weight of the
	#numbers they have misclassified, the weight of their
	#abstentions, and the weight of the ones they have classified
	#correctly.
	d_error = {} #by economist!
	d_w_abstain = {}
	d_w_correct = {}
	d_Z = {}
	d_testpredictions = {}
	d_alpha = {}
	test_dates = []
	train_dates = []
	dates = []

	if KFold == 1:
		with open(filename, "rU") as f:
		    reader = csv.reader(f, delimiter="\t")
		    #a = zip(rows)

		    #prediction_table_T = prediction_table.T
		    #np.random.shuffle(prediction_table_T)
		    #print rows
		    for i, line in enumerate(reader):

		    	#Get weightings by date!
		    	#if i == 0:
		    	#	continue
		    	if i == 0:
		    		continue
		    	if i == 1:
		    		dates.extend(line[0].split(','))
		    		dates.pop(0)
		    		dates = filter(lambda x: len(x)>0 , dates)
		    		avg_weight = 1/len(dates)
		    		for date in dates:
		    			d_weights[date] = avg_weight
		    		#prediction_table[1] = dates
		    		continue

		    	#Get predictions by Economist
		    	predictions = line[0].split(',')
		    	economist = predictions[0]
		    	predictions.pop(0)
		    	#print predictions
		    	predictions = filter(lambda x: x!='' and x!=' ' , predictions)
		    	predictions = [int(j) for j in predictions]
		    	#print predictions
		    	d_predictions[economist] = predictions

	#This is the case where we are dividing the data up into
	#training data and test data.
	if KFold == 2:
		''' 
		Collects the data from the master file
		cleans it in the form of a dictionary of historical estimates.
		Keys are economists
		Values are List of lists containing predictions.
		'''
		with open(filename, "rU") as f:
		    reader = csv.reader(f, delimiter="\t")
		    for i, line in enumerate(reader):

		    	#Get weightings by date!
		    	if i == 0:
		    		continue
		    	if i == 1:
		    		#need to clear commas from names in Sheet
					prediction_table = np.genfromtxt(filename, delimiter=',')
					print prediction_table
					prediction_table[1] = ['nan'] + range(len(prediction_table[2])-1)
					prediction_table = np.delete(prediction_table, (0), axis=0)
					prediction_table = np.delete(prediction_table, (0), axis=1)
					#print prediction_table
					np.random.shuffle(prediction_table.T)
					#print prediction_table
					dates.extend(line[0].split(','))
					dates.pop(0)
					dates = filter(lambda x: len(x)>0 , dates)
					dates = [dates[int(i)] for i in prediction_table[0]]
					#print dates

					# n = len(dates)
					# print "n mod k == 0: ", n % k == 0
					# date_lists = chunkIt(dates,KFold)

					#divide into two lists, training data and test data
					[train_dates,test_dates] = chunkIt(dates,2)

					dates = train_dates
					avg_weight = 1/len(dates)
					for date in dates:
						d_weights[date] = avg_weight
					continue

		    	#Get predictions by Economist
		    	predictions = line[0].split(',')
		    	economist = predictions[0]
		    	# predictions.pop(0)
		    	# predictions = filter(lambda x: x!='' and x!=' ' , predictions)
		    	# predictions = [int(j) for j in predictions]

		    	predictions = prediction_table[i-1]
		    	predictions = [int(j) for j in predictions]
		    	train_predictions = predictions[:int(len(predictions)/2)]
		    	test_predictions = predictions[int(len(predictions)/2):len(predictions)]
		    	d_predictions[economist] = train_predictions
		    	d_testpredictions[economist] = test_predictions

def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out

def invert_bad_economists():
	for economist in d_error.keys():
		if d_error[economist] == d_w_correct[economist]:
			#if an economist has exactly 50% error, his predictions are useless as a weak classifier
			del d_error[economist]
			del d_predictions[economist]
			del d_Z[economist]
			del d_w_correct[economist]
			del d_w_abstain[economist]


		elif d_error[economist] > d_w_correct[economist]:
			#if an economist has >50% error then we consider the inverse of his predictions to be a good weak classifier.
			inverted_predictions = [x*-1 for x in d_predictions[economist]]
			not_economist = str('NOT(')+str(economist)+str(')')

			d_error[not_economist] = d_w_correct[economist]
			d_w_correct[not_economist] = d_error[economist]
			d_w_abstain[not_economist] = d_w_abstain[economist]
			d_predictions[not_economist] = inverted_predictions
			d_Z[not_economist] = d_Z[economist]

			del d_error[economist]
			del d_predictions[economist]
			del d_Z[economist]
			del d_w_correct[economist]
			del d_w_abstain[economist]

def keywithminval(d1):
	temp = d1.copy()
	if len(d_alpha.keys()) > 0:
		print d_alpha
		for economist in d_alpha.keys():
			#print economist in temp.keys()
			del temp[economist]
	v=list(temp.values())
	k=list(temp.keys())
	return k[v.index(min(v))]

def boost(rounds=int):
	global best_weak_classifier
	for iteration in range(1,rounds+1):
		print "Round: ", iteration
		classifier_economist = keywithminval(d_Z)
		print classifier_economist

		#We store the best weak classifier so that we can compare it to the test results of our composite classifier.
		best_weak_classifier = classifier_economist
		
		error = d_error[classifier_economist]
		print "Error: ", error
		Z = d_Z[classifier_economist]
		print "Z: ", Z
		alpha = (math.log(d_w_correct[classifier_economist]/d_error[classifier_economist]))/2
		print "Alpha: ", alpha

		if alpha < 0 or Z >= 1:
			print "No more weak learners"
			break
		d_alpha[classifier_economist] = alpha
		reweight_against(classifier_economist)
		compute_errors()
		compute_D_Z()

def reweight_against(economist=str):
	predictions = d_predictions[economist]
	incorrect_divisor = d_w_abstain[economist] * math.sqrt(d_error[economist]/d_w_correct[economist]) + 2*d_error[economist]
	correct_divisor = d_w_abstain[economist] * math.sqrt(d_w_correct[economist]/d_error[economist]) + 2*d_w_correct[economist]
	abstain_divisor = d_Z[economist]
	#reassign weights to each date
	for i in range(len(dates)):
		day = dates[i]

		if predictions[i] == 1:
			d_weights[day] = d_weights[day]/float(correct_divisor)
			#print 'He was Correct: ', d_weights[day]

		
		elif predictions[i] == -1:
			d_weights[day] = d_weights[day]/float(incorrect_divisor)
			#print 'He was Wrong: ', d_weights[day]

		elif predictions[i] == 0:
			d_weights[day] = d_weights[day]/float(abstain_divisor)

	#recompute W_A, W_C, and W_M (weight of abstains, corrects, and misclassifieds)

def compute_errors():
	for economist in d_predictions.keys():
		curr_error = 0.0
		curr_correct = 0.0
		curr_abstained = 0.0
		for i in range(len(dates)):

			#misclassified case
			if d_predictions[economist][i] == -1:
				curr_error += d_weights[dates[i]]

			#correctly classified case
			elif d_predictions[economist][i] == 1:
				curr_correct+= d_weights[dates[i]]

			#abstaining case
			elif d_predictions[economist][i] == 0:
				curr_abstained += d_weights[dates[i]]

		d_error[economist] = curr_error
		d_w_abstain[economist] = curr_abstained
		d_w_correct[economist] = curr_correct

		if curr_error == 0 or curr_error == 1:
			del d_predictions[economist]
			del d_error[economist]
			del d_w_correct[economist]
			del d_w_abstain[economist]
			if economist in d_Z.keys():
				del d_Z[economist]


def compute_D_Z():
	for economist in d_predictions.keys():
		#Z = W_a + 2*sqrt(W_c * W_m)
		Z = d_w_abstain[economist] + 2*math.sqrt(d_w_correct[economist]*d_error[economist])
		d_Z[economist] = Z
		if not Z < 1:
			if not economist in d_alpha.keys():
				print "Alert! Z >= 1 for ", str(economist)
				print "A,C,M: ",d_w_abstain[economist],d_w_correct[economist],d_error[economist]
				del d_predictions[economist]
				del d_error[economist]
				del d_w_correct[economist]
				del d_w_abstain[economist]
				del d_Z[economist]

def show_HFinal():
	HFinal = "sign("
	for economist in d_alpha.keys():
		HFinal += str(d_alpha[economist]) + " * h(" + str(economist) + ") + "
	HFinal = HFinal[0:len(HFinal)-2]
	print HFinal
	print "Sum of Alpha Values: ", sum(d_alpha.values())


if __name__ == '__main__':
	get_data("ADP CHNG Index.csv",1)
	compute_errors()
	compute_D_Z()
	invert_bad_economists()
	boost(50)
	show_HFinal()