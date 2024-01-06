#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()


# In[2]:


for i in range(5):
    plt.matshow(digits.images[i])


# In[3]:


dir(digits)


# In[4]:


digits.data[0]


# In[5]:


digits.target[0:5]


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2)


# In[8]:


x_train


# In[9]:


x_test


# In[10]:


len(x_train)


# In[11]:


len(x_test)


# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


model=LogisticRegression()


# In[14]:


model


# In[15]:


model.fit(x_train,y_train)


# In[16]:


model.predict(x_test)


# In[17]:


model.score(x_train,y_train)


# In[18]:


model.score(x_test,y_test)


# In[19]:


plt.matshow(digits.images[67])


# In[20]:


digits.target[67]


# In[22]:


model.predict([digits.data[89]])


# In[ ]:




