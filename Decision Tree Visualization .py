#!/usr/bin/env python
# coding: utf-8

# # Decision tree Visualisation

# Importing required libraries

# In[1]:


from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Loading Dataset into the model

# In[2]:


iris = load_iris()
X = iris.data
Y = iris.target
print(X.shape,Y.shape)


# Creation of Dec Tree

# In[3]:


classifier = DecisionTreeClassifier()
classifier.fit(X, Y)

print('Decision Tree Classifer Created')


# Accuracy Check

# In[4]:


Y_Pred = classifier.predict(X)
acc=accuracy_score(Y,Y_Pred)
print(acc)


# For Feeding new data for the purpose of checking

# In[5]:


classifier.predict([[4.9,3.8,4.1,1.5]])


# Visualization of Dec Tree

# In[6]:


fig = plt.figure(figsize=(10, 8))
dec_tree = tree.plot_tree(classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

