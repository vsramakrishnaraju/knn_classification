
#A skeleton for implementing K-nearest Neighbor classifier in Python.
## Author: Venkata

import numpy
import random
import time
import math

trainingFile = "irisPCTraining.txt"
testingFile = "irisPCTesting.txt"
Xtrain = numpy.loadtxt(trainingFile)
n = Xtrain.shape[0]
d = Xtrain.shape[1]-1
print(n, d)

#Testing .....
Xtest = numpy.loadtxt(testingFile)
nn = Xtest.shape[0] # Number of points in the testing data.

tp = 0 #True Positive
fp = 0 #False Positive
tn = 0 #True Negative
fn = 0 #False Negative

#Iterate over all points in testing data
  #For each point find the distances to all the training points.
  #Choose the K points with the smallest distances
  #Assign the class label for the testing point as the majority label of the closes K points.
  #increment TP,FP,FN,TN accordingly, remember the true lable for the ith point is in Xtest[i,d]

#}

#Calculate all the measures required..
k = 21

distance = []
buffer = []
euc_dist_each_test_data =[]

#calulated the distance 

for i in range(0,nn):
    for j in range(0,n):
        squares = (Xtest[i][0]-Xtrain[j][0])**2+(Xtest[i][1]-Xtrain[j][1])**2
        euc_dist_each_test_data = math.sqrt(squares)
        buffer.append(euc_dist_each_test_data)
        euc_dist_each_test_data = []
    distance.append(buffer)
    buffer = []
#print(distance)

    # a1 is first row of the distace matrix (distances from testing data to training data)
for i in range(0,nn-1):
    a1= distance[i]

    # a2 is the last column of the training data i.e lables 
    a2= Xtrain[:,-1]

    # created a map of least k values of distances with lables 

    map = sorted(dict(zip(a1, a2)).items(), key=lambda x: x[0])[:k]

    # numpy conversion

    a = numpy.array(map)

    # lables in and numpy array 

    b = a[:,1]

    # took the most frequent label i.e predicted value 

    def most_frequent(List): 
        counter = 0
        num = List[0] 

        for i in List: 
            curr_frequency = List.count(i) 
            if(curr_frequency> counter): 
                counter = curr_frequency 
                num = i 

        return num 

    predicted = most_frequent(b.tolist())

    # took the actual values from test data 

    actual = Xtest[i-1][d]

    # comparision of predicted with actual values of the lables 

    if actual == 1.0 and predicted == 1.0:
        tp=tp+1
    if actual == -1.0 and predicted == 1.0:
        fp=fp+1
    if actual == -1.0 and predicted == -1.0:
        tn=tn+1
    if actual == 1.0 and predicted == -1.0:
        fn=fn+1

accuracy = (tp+tn)/(tp+fp+tn+fn)
sensitivity = tp/(tp+fn)
specificity = tn/(fp+tn)
precision = tp/(tp+fp)


print("k",k)
print("accuracy",accuracy)
print("sensitivity",sensitivity)
print("specificity",specificity)
print("precision",precision)