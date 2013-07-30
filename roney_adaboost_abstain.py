from __future__ import division #makes everything a float
import csv
import numpy as np
import operator
import math
import random
import glob

#def reshuffle()
#def train_on_future_economists_only




def get_data(filename=str):
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

	global original_dates
	global test_dates
	global train_dates
	global d_testpredictions
	global prediction_table
	global d_HofX
	global test_date
	global used_testdate
	global wins
	global losses

	losses = []
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



	''' 
	Collects the data from the master file
	cleans it in the form of a dictionary of historical estimates.
	Keys are economists
	Values are List of lists containing predictions.
	'''
	#need to clear commas from names in Sheet
	prediction_table = np.genfromtxt(filename, dtype=None, delimiter=',')
	#Remove empty row at top
	prediction_table = np.delete(prediction_table, (0), axis=0)
	prediction_table[0] = ['Dates'] + range(len(prediction_table[0])-1)

def reshuffle_data():
	np.random.shuffle(prediction_table.T)

def organize_by_testDate(this_date=int):
	global indx
	global dates
	global test_date
	dates = range(len(prediction_table[0])-1)

	test_date = this_date
	indx = dates.index(this_date)
	dates.pop(indx)

	d_predictions.clear()
	d_testpredictions.clear()
	d_Z.clear()
	d_weights.clear()
	d_error.clear()
	d_alpha.clear()
	d_w_correct.clear()
	d_w_abstain.clear()

	avg_weight = 1/len(dates)
	for date in dates:
		d_weights[date] = avg_weight

	#consider the string "Actual Direction" at the beginning
	actual_direction = prediction_table[1][indx+1]

	#Get predictions by Economist
	#Start at row 2 because we don't want to include Actual Direction!
	for row in range(2, prediction_table.shape[0]):
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

		#We take everything except the specified item as the training data
		test_prediction = predictions[indx]
		predictions.pop(indx)
		train_predictions = predictions

		d_predictions[economist] = train_predictions
		d_testpredictions[economist] = test_prediction

	#We only want to train on the economists who have existing predictions on the test data. Otherwise it's
	#useless to us.
	for economist in d_testpredictions.keys():
		#If they didn't make a prediction then we have them listed as 0.
		if int(d_testpredictions[economist]) == 0:
			del d_predictions[economist]
			del d_testpredictions[economist]
	compute_errors()
	compute_D_Z()
	invert_bad_economists()

def invert_bad_economists():
	for economist in d_error.keys():
		if (d_error[economist] == d_w_correct[economist]) or (d_error[economist] == 0.0) or (d_error[economist] == 1):
			#if an economist has exactly 50% error, his predictions are useless as a weak classifier
			#Also AdaBoost only works on noisy data. Economists who are perfect or 100% (or 0%) in their
			#predictions are not considered noisy, and we can't use them.
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


def delete_Noiseless_Economists():
	#Manipulating a dictionary while it is being iterated through: Bad practice! Have to run it twice to make it work :-/
	for economist in d_error.keys():
		if (d_error[economist] == d_w_correct[economist]) or (d_error[economist] == 0.0) or (d_error[economist] == 1):
			#if an economist has exactly 50% error, his predictions are useless as a weak classifier
			#Also AdaBoost only works on noisy data. Economists who are perfect or 100% (or 0%) in their
			#predictions are not considered noisy, and we can't use them.
			del d_error[economist]
			del d_predictions[economist]
			del d_Z[economist]
			del d_w_correct[economist]
			del d_w_abstain[economist]
			del d_testpredictions[economist]

def boost(rounds=int):
	global best_weak_classifier
	delete_Noiseless_Economists()
	print "Test Date: ",test_date
	print file1
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

		if (alpha < 0) or (Z >= 1) or (len(d_alpha.keys())>=len(d_Z.keys())):
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

		#print curr_error
		d_error[economist] = curr_error
		d_w_abstain[economist] = curr_abstained
		d_w_correct[economist] = curr_correct

		if curr_error == 0 or curr_error == 1:
			#print economist
			del d_predictions[economist]
			del d_testpredictions[economist]
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
	actual_direction = int(prediction_table[1][indx+1])
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
	print "H_Final: ", h_fin
	print "Actual Direction:", actual_direction
	if actual_direction*h_fin>0:
		print "Correct"
		wins.append(test_date)
	else:
		print "Wrong"
		losses.append(test_date)

def workhorsePipeline():
	d_results = {}
	files = glob.glob('*.csv')
	global file1
	for file1 in files:
		if file1 == "RSTAXAG% Index.csv":
			continue
		get_data(file1)
		for w in range(len(prediction_table[0])-1):
			organize_by_testDate(w)
			boost(50)
			show_HFinal()
			test_against_Actuals()

			d_predictions.clear()
			d_testpredictions.clear()
			d_Z.clear()
			d_weights.clear()
			d_error.clear()
			d_alpha.clear()
			d_w_correct.clear()
			d_w_abstain.clear()
		print "Wins:", wins
		print "Losses:", losses
		print "Win %age",len(wins)/(len(wins)+len(losses))
		d_results[file1] = len(wins)/(len(wins)+len(losses))
	for k in d_results.keys():
		print k,d_results[k]


def test_New_Data():
	get_data("CONCCONF Index.csv")
	organize_by_testDate(last_column)
	boost(50)
	show_HFinal()
	h_fin = 0.0
	for economist in d_alpha.keys():
		if d_testpredictions[economist] == 0:
			continue
		elif d_testpredictions[economist] == 1:
			h_fin += 1*d_alpha[economist]
		elif d_testpredictions[economist] == -1:
			h_fin += -1*d_alpha[economist]
	print "H_Final: ", h_fin


if __name__ == '__main__':
	#workhorsePipeline()
	test_New_Data()
