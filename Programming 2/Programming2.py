# Thomas Pollard
# CS445 - Programming2.py
import numpy as np
import math
import sys
from sklearn.model_selection import train_test_split

# Load the data
myData = np.genfromtxt('spambase.data', delimiter=',')

# Split the data into a training and test set
dataSize = myData.shape[0]
trainSet, testSet = train_test_split(myData, test_size=int(dataSize/2))

# Split each set into spam and legit
trainSpam = trainSet[trainSet[:,-1]==1]
trainLegit = trainSet[trainSet[:,-1]==0]
testSpam = testSet[testSet[:,-1]==1]
testLegit = testSet[testSet[:,-1]==0]

# Find the stats of the two sets for debugging purposes
trainSize = trainSet.shape[0]
trainSpamCount = trainSpam.shape[0]
trainLegitCount = trainLegit.shape[0]
trainSpamPrior = trainSpamCount/trainSize
trainLegitPrior = trainLegitCount/trainSize
    
testSize = testSet.shape[0]
testSpamCount = testSpam.shape[0]
testLegitCount = testLegit.shape[0]
testSpamPrior = testSpamCount/testSize
testLegitPrior = testLegitCount/testSize


# Print set stats for verification and debug purposes
def printSetStats():
    # Print the training set info

    print("Number of data in train set:", trainSize)
    print("Number of spams in train set:",trainSpamCount)
    print("Number of legitimate in train set:",trainLegitCount)


    print("Prior of spams in train set:", trainSpamPrior)
    print("Prior of legit in train set:", trainLegitPrior)

    print("\n")
    # Print the test set info
    print("Number of data in test set:", testSize)
    print("Number of spams in test set:",testSpamCount)
    print("Number of legitimate in test set:",testLegitCount)

    print("Prior of spams in test set:", testSpamPrior)
    print("Prior of legit in test set:", testLegitPrior)

    # Print the mean and std devs of the spam and legit training sets
    print("Featurewise mean of the spam training set:")
    for n in range(len(spamFeatureMeans)):
        print(n, ":" , spamFeatureMeans[n])
        
    print("Featurewise std dev of the spam training set:")
    for n in range(len(spamFeatureStdDevs)):
        print(n, ":" , spamFeatureStdDevs[n])
        
    print("Featurewise mean of the legit training set:")
    for n in range(len(legitFeatureMeans)):
        print(n, ":" , legitFeatureMeans[n])
        
    print("Featurewise std dev of the legit training set:")
    for n in range(len(legitFeatureStdDevs)):
        print(n, ":" , legitFeatureStdDevs[n])

def probXGivenSpam (i,inputData):
    stdDev = spamFeatureStdDevs[i]
    mean = spamFeatureMeans[i]
    x = inputData
    result = (1/(math.sqrt(2 * math.pi)* stdDev))*math.exp(-((x-mean)**2/(2*stdDev**2)))
    if result == 0:
        return sys.float_info.epsilon
    else:
        return result

def probXGivenLegit (i,inputData):
    stdDev = legitFeatureStdDevs[i]
    mean = legitFeatureMeans[i]
    x = inputData
    result = (1/(math.sqrt(2 * math.pi)* stdDev))*math.exp(-((x-mean)**2/(2*stdDev**2)))
    if result == 0:
        return sys.float_info.epsilon
    else:
        return result

def classify(x,i):
    # Find probability of spam
    classProb = np.empty(len(x)-1)
    for n in range(len(classProb)):
        classProb[n] = math.log(probXGivenSpam(n,x[n]))

    spamProb = math.log(spamPrior) + np.sum(classProb)
    
    # Find probability of legit
    for n in range(len(classProb)):
        classProb[n] = math.log(probXGivenLegit(n,x[n]))
        
    legitProb = math.log(legitPrior) + np.sum(classProb)

    if max(spamProb,legitProb) == spamProb:
        return 1
    else:
        return 0
    
        
# Compute prior probability for each class in the training data
spamPrior = trainSpamPrior
legitPrior = trainLegitPrior

# Compute the featurewise mean and std dev on the spams in the training set
spamFeatureMeans = np.mean(trainSpam, axis=0)
spamFeatureStdDevs = np.std(trainSpam, axis=0)
spamFeatureStdDevs[spamFeatureStdDevs == 0] = 0.0001

# Compute the featurewise mean and std dev on the legitimates in the training set
legitFeatureMeans = np.mean(trainLegit, axis=0)
legitFeatureStdDevs = np.std(trainLegit, axis=0)
legitFeatureStdDevs[legitFeatureStdDevs == 0] = 0.0001


#printSetStats()
obj = trainSpam
result = np.array([classify(obj[i],i) for i in range(obj.shape[0])])
print("classify trainSpam returned :\n",np.sum(result), "spams")


obj = trainLegit
result = np.array([classify(obj[i],i) for i in range(obj.shape[0])])
print("classify trainLegit returned :\n",np.sum(result), "spams")

# Confusion matrix with row/column 0 is legit and row/column 1 is spam
confusionMatrix = [[0,0],[0,0]]
obj = testSpam
result = np.array([classify(obj[i],i) for i in range(obj.shape[0])])
print("classify testSpam returned :\n",np.sum(result), "spams")
confusionMatrix[1][1] = np.sum(result)
confusionMatrix[1][0] = testSpam.shape[0]-confusionMatrix[1][1]


obj = testLegit
result = np.array([classify(obj[i],i) for i in range(obj.shape[0])])
print("classify testLegit returned :\n",np.sum(result), "spams")
confusionMatrix[0][1] = np.sum(result)
confusionMatrix[0][0] = testLegit.shape[0]-confusionMatrix[0][1]

print("Confusion Matrix:\n", np.array(confusionMatrix))
