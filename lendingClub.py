import sys
import csv
import numpy
from sklearn import tree

# local variables
attributes = list()
allData = list()
training = list()
test = list()
allClassifications = list()
trainingClassifications = list()
testClassifications = list()
relevantAttributes = list()
classification = 16
correctPredictions = float()
predictionsCounter = dict()
correctPredictionsCounter = dict()
finalPredictionsCounter = dict()

# parameters
trainingSize = 1000

def findFeatureIndex(featureName, data, attributes):
	featureIndex = None
	for i in range(len(attributes)):
		if attributes[i].strip() == featureName:
			featureIndex = i
	return featureIndex

# reading csv file
with open('LoanStats.csv') as csvfile:
	csv.field_size_limit(sys.maxsize)
	reader = csv.reader(csvfile, delimiter="|", quotechar='|')
	next(reader)
	attributesCSV = next(reader)
	attributes = attributesCSV[0].split(",")

	# creating training and test lists
	rowNum = 0
	for row in reader:
		tempData = list()
		for segment in row:
			tempData += segment.split(",")

		# bypass the loan_status bug
		if tempData[classification] == "Mar-16":
			continue
		if tempData[classification] not in ["Charged Off", "Fully Paid"]:
			continue

		# bypass annual_inc bug
		try:
		    float(tempData[13])
		except ValueError:
		    continue

		allData.append(tempData)
		rowNum += 1

# feature indices
funded_amnt = findFeatureIndex("loan_amnt", allData, attributes)
int_rate = findFeatureIndex("int_rate", allData, attributes)
installment = findFeatureIndex("installment", allData, attributes)
grade = findFeatureIndex("grade", allData, attributes)
sub_grade = findFeatureIndex("sub_grade", allData, attributes)
annual_inc = findFeatureIndex("annual_inc", allData, attributes)

# editing features
for data in allData:

	# int_rate
	data[int_rate] = data[int_rate][:-1]

	# grade
	data[grade] = ord(data[grade]) - 65

	# sub_grade
	data[sub_grade] = (ord(data[sub_grade][0]) - 65) * 10 + int(data[sub_grade][1]) * 2


# picking relevant attributes
relevantAttributes = [sub_grade]
print relevantAttributes

# reorganizing data
for i in range(len(allData)):
	tempSample = list()
	for index in relevantAttributes:
		tempSample.append(allData[i][index])
	allClassifications.append(allData[i][classification])
	allData[i] = tempSample

# splitting data
training = allData[:trainingSize]
trainingClassifications = allClassifications[:trainingSize]
test = allData[trainingSize:]
testClassifications = allClassifications[trainingSize:]

# fitting decision tree
print "fitting decision tree..."
clf = tree.DecisionTreeClassifier()
clf = clf.fit(training, trainingClassifications)

# analyzing tree performance
validPredicions = 0
for i in range(len(test)):
	if test[i][0] != "":
		validPredicions += 1
		prediction = clf.predict(numpy.reshape(test[i], (1, -1)))
		if prediction == [testClassifications[i]]:
			correctPredictions += 1
			if prediction[0] not in correctPredictionsCounter:
				correctPredictionsCounter[prediction[0]] = 0
			correctPredictionsCounter[prediction[0]] += 1
		if prediction[0] not in predictionsCounter:
			predictionsCounter[prediction[0]] = 0
		predictionsCounter[prediction[0]] += 1
for prediction in predictionsCounter:
	if prediction not in correctPredictionsCounter:
		correctPredictionsCounter[prediction] = 0
	finalPredictionsCounter[prediction] = float(correctPredictionsCounter[prediction]) / predictionsCounter[prediction]

print set(allClassifications)
print predictionsCounter
print correctPredictionsCounter
print finalPredictionsCounter
print len(test), "test entries |", len(training), "training entries |", len(test) + len(training), "total entries"
print correctPredictions * 100 / len(test), "percent accuracy |", validPredicions - len(test), "invalid tests"
print ""