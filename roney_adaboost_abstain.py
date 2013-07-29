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
	global prediction_table
	global d_HofX
	global test_date
	global used_testdate
	global wins

	wins = []
	used_testdate = []
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
			for i, line in enumerate(reader):

				#Get weightings by date!
				if i == 0:
					continue
				if i == 1:
					dates.extend(line[0].split(','))
					dates.pop(0)
					dates = filter(lambda x: len(x)>0 , dates)
					avg_weight = 1/len(dates)
					for date in dates:
						d_weights[date] = avg_weight
					continue

				#Get predictions by Economist
				predictions = line[0].split(',')
				economist = predictions[0]
				predictions.pop(0)
				predictions = filter(lambda x: x!='' and x!=' ' , predictions)
				predictions = [int(j) for j in predictions]
				d_predictions[economist] = predictions

			#This is the case where we are dividing the data up into
	#training data and test data.

	#I think we should forget the 2-Fold CV, and do (n-1) points for training data and 1 test point.
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
					prediction_table = np.genfromtxt(filename, dtype=None, delimiter=',')
					prediction_table[1] = ['nan'] + range(len(prediction_table[2])-1)
					prediction_table = np.delete(prediction_table, (0), axis=0)
					prediction_table = np.delete(prediction_table, (0), axis=1)
					np.random.shuffle(prediction_table.T)
					dates.extend(line[0].split(','))
					dates.pop(0)
					dates = filter(lambda x: len(x)>0 , dates)
					dates = [dates[int(i)] for i in prediction_table[0]]


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

	if KFold == 3:
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
					prediction_table = np.genfromtxt(filename, dtype=None, delimiter=',')
					prediction_table[1] = ['nan'] + range(len(prediction_table[2])-1)
					print prediction_table[1]
					print prediction_table
					prediction_table = np.delete(prediction_table, (0), axis=0)
					prediction_table = np.delete(prediction_table, (0), axis=1)
					print prediction_table
					np.random.shuffle(prediction_table.T)

					dates.extend(line[0].split(','))
					dates.pop(0)
					dates = filter(lambda x: len(x)>0 , dates)
					dates = [dates[int(x)] for x in prediction_table[0]]

					test_date = dates[-1]
					dates.pop()

					test_table = prediction_table.T[-1]
					prediction_table = np.delete(prediction_table,(len(prediction_table[0])-1), axis=1)

					avg_weight = 1/len(dates)
					#print len(dates)
					for date in dates:
						d_weights[date] = avg_weight
					continue

				#Get predictions by Economist
				predictions = line[0].split(',')
				economist = predictions[0]
				predictions = prediction_table[i-1]
				predictions = [int(j) for j in predictions]
				train_predictions = predictions[:len(predictions)-1]
				test_prediction = int(predictions[-1])
				d_predictions[economist] = train_predictions
				d_testpredictions[economist] = test_prediction

	    	for economist in d_testpredictions.keys():
				if int(d_testpredictions[economist]) == 0:
					del d_predictions[economist]
					del d_testpredictions[economist]


	if KFold == 4:
		''' 
		Collects the data from the master file
		cleans it in the form of a dictionary of historical estimates.
		Keys are economists
		Values are List of lists containing predictions.
		'''
		#need to clear commas from names in Sheet
		prediction_table = np.genfromtxt(filename, dtype=None, delimiter=',')
		prediction_table[1] = ['Dates'] + range(len(prediction_table[2])-1)
		#print prediction_table[1]
		#print prediction_table
		prediction_table = np.delete(prediction_table, (0), axis=0)
		#prediction_table = np.delete(prediction_table, (0), axis=1)
		#print prediction_table
		np.random.shuffle(prediction_table.T)

		dates = prediction_table[0]
		dates = filter(lambda x: len(x)>0 , dates)
		for day in dates:
			if not day.isdigit():
				dates.pop(dates.index(day))

		dates = [int(x) for x in dates]

		test_date = dates[-1]

		test_table = prediction_table.T[-1]
		avg_weight = 1/len(dates)
		for date in dates:
			d_weights[date] = avg_weight

		#Get predictions by Economist
		for row in range(1, prediction_table.shape[0]):
			predictions = []
			economist = ''
			for item in prediction_table[row]:
				#we consider '-1' the shortest string which is a direction number
				if len(item)<=2:
					#notice that this is keeping in order!
					predictions.append(int(item)) #add it to the predictions as an integer
				else:
					#A string that is not a number MUST be the Economist's name
					economist = item
			#We take everything except the last item as the training data
			train_predictions = predictions[:len(predictions)-1]
			#We use only the last column of the prediction_table as test data. 
			test_prediction = int(predictions[-1])
			d_predictions[economist] = train_predictions
			d_testpredictions[economist] = test_prediction

		#We only want to train on the economists who have existing predictions on the test data. Otherwise it's
		#useless to us.
		for economist in d_testpredictions.keys():
			#If they didn't make a prediction then we have them listed as 0.
			if int(d_testpredictions[economist]) == 0:
				del d_predictions[economist]
				del d_testpredictions[economist]
		print d_testpredictions.keys()




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
			del d_testpredictions[economist]


		elif d_error[economist] > d_w_correct[economist]:
			#if an economist has >50% error then we consider the inverse of his predictions to be a good weak classifier.
			inverted_predictions = [x*-1 for x in d_predictions[economist]]
			not_economist = str('NOT(')+str(economist)+str(')')

			d_error[not_economist] = d_w_correct[economist]
			d_w_correct[not_economist] = d_error[economist]
			d_w_abstain[not_economist] = d_w_abstain[economist]
			d_predictions[not_economist] = inverted_predictions
			d_testpredictions[not_economist] = d_testpredictions[economist]*-1
			d_Z[not_economist] = d_Z[economist]

			del d_error[economist]
			del d_predictions[economist]
			del d_Z[economist]
			del d_w_correct[economist]
			del d_w_abstain[economist]
			del d_testpredictions[economist]

def keywithminval(d1):
	temp = d1.copy()
	if len(d_alpha.keys()) > 0:
		for economist in d_alpha.keys():
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
		reweight_against2(classifier_economist)
		compute_errors2()
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

def reweight_against2(economist=str):
	predictions = d_predictions[economist]
	incorrect_divisor = d_w_abstain[economist] * math.sqrt(d_error[economist]/d_w_correct[economist]) + 2*d_error[economist]
	correct_divisor = d_w_abstain[economist] * math.sqrt(d_w_correct[economist]/d_error[economist]) + 2*d_w_correct[economist]
	abstain_divisor = d_Z[economist]
	#reassign weights to each date
	for i in range(len(dates)-1):
		day = dates[i]

		if predictions[i] == 1:
			d_weights[day] = d_weights[day]/float(correct_divisor)
			#print 'He was Correct: ', d_weights[day]

		elif predictions[i] == -1:
			d_weights[day] = d_weights[day]/float(incorrect_divisor)
			#print 'He was Wrong: ', d_weights[day]

		elif predictions[i] == 0:
			d_weights[day] = d_weights[day]/float(abstain_divisor)

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

def compute_errors2():
	for economist in d_predictions.keys():
		curr_error = 0.0
		curr_correct = 0.0
		curr_abstained = 0.0
		for i in range(len(dates)-1):
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

def test_against_Actuals():
	actual_direction = int(prediction_table[1][-1])
	h_fin = 0.0

	for economist in d_alpha.keys():
		if actual_direction == 0:
			print "Actual == Consensus"
			print test_date
			break
		elif actual_direction == 1:
			if d_testpredictions[economist] == -1:
				h_fin += -1*d_alpha[economist]
			elif d_testpredictions[economist] == 1:
				h_fin += 1*d_alpha[economist]
		elif actual_direction == -1:
			if d_testpredictions[economist] == -1:
				h_fin += 1*d_alpha[economist]
			elif d_testpredictions[economist] == 1:
				h_fin += -1*d_alpha[economist]
	print "H_fin: ", h_fin
	print "Actual Direction:", actual_direction



if __name__ == '__main__':
	get_data("CONCCONF Index.csv",4)
	print "posterity"
	print d_predictions
	print d_testpredictions
	print prediction_table[1]
	compute_errors2()
	compute_D_Z()
	invert_bad_economists()
	boost(50)
	show_HFinal()
	test_against_Actuals()


