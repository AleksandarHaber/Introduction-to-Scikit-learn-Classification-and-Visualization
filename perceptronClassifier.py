# -*- coding: utf-8 -*-
"""
Created on Thu May 26 20:54:12 2022

@author: ahaber
"""
# Introduction to Classification Problems using Scikit-learn library

# we import data sets
from sklearn import datasets
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from visualizeFunctions import visualizeClassificationAreas

# load the data set
irisDataSet =datasets.load_iris()

# explore the data set
# we have 4 features
irisDataSet['feature_names']
# we have 3 targets - 3 classes
irisDataSet['target_names']
# feature data set : number of samples \times number of features
irisDataSet['data']
irisDataSet['target']

# extract only two features for classification
Xfeatures=irisDataSet['data'][:,[2,3]]
YtargetClasses=irisDataSet['target']

# we perform multi-class classification -since the target set has more than two classes
# however, we will use One-vs-Rest (OvR) heuristic method. 
# OvR uses a heuristic method where the multi-class problem is decomposed into series 
# of binary classification problems. For example, if we have 3 classes {0,1,2}, 
# then we train the following classifiers:
# Binary classifier 1: class 0 is class A, and classes 1,2 are the class B 
# Binary classifier 2: class 1 is class A, and classes 0,2 are the class B 
# Binary classifier 3: class 2 is class A, and classes 1,2 are the class B 

XfeaturesTrain, XfeaturesTest, YtargetClassesTrain, YtargetClassesTest = train_test_split(Xfeatures,
YtargetClasses,test_size=0.2, random_state=1, stratify=YtargetClasses)

# arguments:
# 1. features, 2. targets, 3. test/train size ratio, 
# 4.random_state=1 make sure that the results are repeatable every time we run the function
# 5. stratify=y - stratified sampling to make sure that the proportio of samples in train 
# and test data sets correspond to the proportion of samples of classes in the original 
# YtargetClassesTest data set. That is empirical distributions of classes in train and 
# test data sets have to mach the empirical distribution of classes in the original data set

# let us test that the stratified sampling is achieved 
np.bincount(YtargetClasses)
np.bincount(YtargetClassesTrain)
np.bincount(YtargetClassesTest)

# scale the data to improve the learning performance
# basically, the standard scaler standardizes the data by estimating the 
# mean and the standard deviation, and by transforming the data by using these 
# estimates
standardScaler = StandardScaler()
standardScaler.fit(Xfeatures)
# here we transform the data
XfeaturesTrainScaled=standardScaler.transform(XfeaturesTrain)
XfeaturesTestScaled=standardScaler.transform(XfeaturesTest)

# initialize the perceptronClassifier
perceptronClassifier=Perceptron(eta0=0.1,random_state=1)
# eta0=0.1 learning rate
# random_state=1, ensure that the results are reproducible due to initial random shuffling of the data

# learn the model
perceptronClassifier.fit(XfeaturesTrainScaled,YtargetClassesTrain)

# predict the classes
predictedClassesTest=perceptronClassifier.predict(XfeaturesTestScaled)

# check the performance 
numberOfMisclassification= (predictedClassesTest!=YtargetClassesTest).sum()

# percentage of misclassification
misclassificationPercentage=(numberOfMisclassification/(YtargetClassesTest.size))*100

# plot the decision regions
visualizeClassificationAreas(perceptronClassifier,XfeaturesTrainScaled,YtargetClassesTrain,XfeaturesTestScaled,YtargetClassesTest)
