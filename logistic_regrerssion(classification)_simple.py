#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:/Users/Darshan/Downloads/insurance_data.csv")


# In[3]:


df


# In[4]:


plt.scatter(df.age,df.bought_insurance)


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


x_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,test_size=0.1) #which variable is used to predict our output that is two dimensional


# In[7]:


x_test


# In[8]:


x_train


# In[9]:


from sklearn.linear_model import LogisticRegression


# In[10]:


model= LogisticRegression()
model.fit(x_train,y_train)


# 

# In[11]:


model.predict(x_test)


# In[12]:


model.score(x_train,y_train)


# In[13]:


model.score(x_test,y_test)


# In[14]:


#sigmoid function


# In[15]:


import math
def sigmoid(x):
    return 1/(1+math.exp(-x))


# In[16]:


def predict_function(age):
    z=0.042*age-1.53
    y=sigmoid(z)
    return y


# In[17]:


age=45
predict_function(age)


# In[18]:


age=25
predict_function(age)


# In[ ]:




