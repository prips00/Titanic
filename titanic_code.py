#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# In[2]:


from warnings import filterwarnings
filterwarnings(action = "ignore")


# In[3]:


#Data Reading

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[4]:


#Data exploration

print(train.head())

print("-----------------------------------------------------------")

print(test.head())


# In[5]:


#Shape of dataset

print(train.shape)
print(test.shape)


# In[6]:


#Null value checks

print(train.isnull().sum())
print("*******************************")
print(test.isnull().sum())


# In[7]:


#Description of dataset

print(train.describe())
print("********************************************************")
print(test.describe())


# In[8]:


#Plotting the data relations

plt.figure(1)
train.loc[train['Survived'] == 1 ,"Pclass" ].value_counts().sort_index().plot.bar()
plt.title("Bar graph of people who survived based on their ticket class")


# In[9]:



plt.figure(2)
train.loc[train['Survived'] == 0 ,"Pclass" ].value_counts().sort_index().plot.bar()
plt.title("Bar graph of people who didn't survived based on their ticket class")


# In[10]:



plt.figure(3)
plt.figure(figsize=(17,6))
train.loc[train['Survived'] == 1 ,"Age" ].value_counts().sort_index().plot.bar()
plt.title("Bar graph of people who survived based on their age")


# In[11]:


plt.figure(4)
plt.figure(figsize=(17,6))
train.loc[train['Survived'] == 0 ,"Age" ].value_counts().sort_index().plot.bar()
plt.title("Bar graph of people who didn't survived based on their age")


# In[12]:


plt.figure(5)
plt.figure(figsize=(33,8))
train.loc[train['Survived'] == 1 ,"Fare" ].value_counts().sort_index().plot.bar()
plt.title("Bar graph of people who survived based on their fare")


# In[13]:


plt.figure(4)
plt.figure(figsize=(33,8))
train.loc[train['Survived'] == 0 ,"Fare" ].value_counts().sort_index().plot.bar()
plt.title("Bar graph of people who didn't survived based on their fare")


# In[14]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis("equal") 

l = ["c = cherbourg", "s = southampton" , "q =  queenstown" ]
s = [0.546464,0.23434 , 0.5354343]
ax.pie(s,labels = l,autopct = "%1.2f%%")


# In[15]:


#Understanding hidden patterns

train[ ["Pclass","Survived"]].groupby(["Pclass"] , as_index = False ).mean().sort_values(by="Survived", ascending = False)


# In[16]:


train[ ["SibSp","Survived"]].groupby(["SibSp"] , as_index = False ).mean().sort_values(by="Survived", ascending = False)


# In[17]:


#Removing unneccesary data

train = train.drop(["Ticket"], axis = 1)
test = test.drop(["Ticket"], axis = 1 )
train = train.drop(["Name"], axis = 1)
test = test.drop(["Name"], axis = 1 )
train = train.drop(["Cabin"], axis = 1)
test = test.drop(["Cabin"], axis = 1 )


# In[18]:


#Selecting important features

column_train = [ "Age" , "Pclass" ,"SibSp" ,"Parch" ,"Fare","Sex","Embarked"]


X = train[column_train]
Y = train["Survived"]


# In[19]:


#Data pre-processing

X["Age"] = X["Age"].fillna( X["Age"].median())

X["Age"].isnull().sum()

d = {"male" : 0 ,"female":1}

X['Sex'] = X["Sex"].apply( lambda x:d[x])

X.head()

X["Embarked"] = X["Embarked"].fillna( "C")


e = {"S" : 2,"Q":1,"C":0}

X["Embarked"] = X["Embarked"].apply(lambda x:e[x] )


# Model Building and Training

# In[20]:


from sklearn.model_selection import train_test_split

X_train ,X_test ,Y_train ,Y_test = train_test_split(X,Y, test_size = 0.3,random_state = 7)


# In[21]:


#Logistic Regression (Model 1)

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

model1 = LogisticRegression() 
model1.fit(X_train,Y_train)

y_pred1 = model1.predict(X_test)
Accuracy1 = accuracy_score(Y_test,y_pred1) * 100 


# In[22]:


#Standard Vector Machine (Model 2)

from sklearn.svm import SVC 

model2 = SVC()
model2.fit(X_train,Y_train)

y_pred2 = model2.predict(X_test)
Accuracy2 = accuracy_score(Y_test,y_pred2) * 100 


# In[23]:


#Naive Bayes algorithm (Model 3)

from sklearn.naive_bayes import GaussianNB 

model3 = GaussianNB()
model3.fit(X_train,Y_train)

y_pred3 = model3.predict(X_test)
Accuracy3 = accuracy_score(Y_test,y_pred3) * 100 


# In[24]:


#Decision Tree Algorithm (Model 4)

from sklearn.tree import DecisionTreeClassifier 

model4 =DecisionTreeClassifier( criterion= "entropy" ,random_state = 7)
model4.fit(X_train,Y_train)

y_pred4 = model4.predict(X_test)
Accuracy4= accuracy_score(Y_test,y_pred4) * 100 


# In[25]:


print("Accuracy of the following models are as follows:")
print(" ")
print("Model 1 -> ",Accuracy1)
print("Model 2 -> ",Accuracy2)
print("Model 3 -> ",Accuracy3)
print("Model 4 -> ",Accuracy4)

