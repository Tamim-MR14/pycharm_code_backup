from setuptools.command.test import test
from sklearn import tree
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

trainDocs = pd.read_csv("Youtube01-Psy",header=0)
#drop some of the columns
cols = ['COMMENT_ID','AUTHOR','DATE']
trainDocs = trainDocs.drop(cols,axis=1)
print(trainDocs)
trainDocs = ["Hulk like fire fire","Thor like water"]
ClassNames = ["Hulk","Thor"]
YTrain = [0,1]

testDocs = ["Hulk Smash!","Thor beat Loki"]
YTest = [0,1]