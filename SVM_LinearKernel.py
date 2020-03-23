# Created by: Gayathri Krishnamoorthy
# Updated: 03-23-2020

# Support Vector Machine using Linear Kernel for fashion Mnist data is implemented here.
# The classifier is trained for different values of C parameters.
# It is coded in python version 3.6.


import numpy as np
import csv
import os
from math import floor
import matplotlib.pyplot as plt
from numpy import linalg
import pandas as pd
from pprint import pprint

from tqdm import tqdm
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

## get fashionmnist data
def ReadData(path):
    with open(path, encoding='utf-8-sig') as fn:
        csvreader = csv.reader(fn)
        next(csvreader) ### SKIP COLUMN HEADS ###

        label_len = 10
        feat_len = 784
        x = []
        y = []
        for row in csvreader:
            x.append(list(float(row[i]) for i in range(1,feat_len+1)))
            y.append(int(row[0]))
        x = np.array(x)
        y = np.array(y)

    return x, y

## split to training and test features
x_train, y_train = ReadData('./fashionmnist/fashion-mnist_train.csv')
x_test, y_test = ReadData('./fashionmnist/fashion-mnist_test.csv')

x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.2)
print('doodoo')

# print accuracy on training and test data # number of iterations 
# for different C values 

acc = []
C_val = [10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3, 10**4]
for C in tqdm(C_val):
    linSVM = LinearSVC(C = C)
    linSVM.fit(x_train,y_train)
    pred_train_y = linSVM.predict(x_train_split)
    pred_val_y = linSVM.predict(x_val)
    pred_test_y = linSVM.predict(x_test)
    acc_train = accuracy_score(y_train_split, pred_train_y)
    acc_val = accuracy_score(y_val, pred_val_y)
    acc_test = accuracy_score(y_test, pred_test_y)
    #num_vec = linSVM.n_support_ ### Not supported by LinearSVC (svm.SVC has)
    tmp = [C, acc_train, acc_val, acc_test]
    acc.append(tmp)

print(acc)
acc_array = np.asarray(acc)
row = np.unravel_index(acc_array[:,2].argmax(),acc_array[:,2].shape)[0] # returns row with max validation accuracy
print(row)
cmax = acc_array[row][0]
print(cmax)

## select the best C and find testing accuracy and 10*10 confusion matrix

linSVM = LinearSVC(C = cmax)
linSVM.fit(x_train,y_train)
pred_y = linSVM.predict(x_test)
best_acc = accuracy_score(y_test,pred_y)  # Accuracy for best C on full training data
conf = confusion_matrix(y_test,pred_y)
print('Linear Acc:', best_acc)
print(conf)

poly_acc = []

for n in range(2,5):
    polySVM = svm.SVC(C = cmax, kernel='poly', degree=n)
    polySVM.fit(x_train,y_train)
    pred_y = polySVM.predict(x_test)
    tmp_acc = accuracy_score(y_test,pred_y)
    poly_acc.append(tmp_acc) # Accuracies for each polynomial kernel run

print('c_max:', cmax, 'lin acc:', best_acc, 'poly 2 acc:', poly_acc[0], 'poly 3 acc:', poly_acc[1], 'poly 4 acc', poly_acc[2])
print(' C,train acc, validation acc, test acc')
for line in acc:
    print(line)



