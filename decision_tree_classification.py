#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("C:/Users/Darshan/Downloads/salaries.csv")
df.head()


# In[3]:


inputs = df.drop('salary_more_then_100k',axis='columns')


# In[4]:


target = df['salary_more_then_100k']


# In[5]:


from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


# In[6]:


inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])


# In[7]:


inputs


# In[8]:


inputs_n = inputs.drop(['company','job','degree'],axis='columns')


# In[9]:


inputs_n


# In[10]:


target


# In[11]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[12]:


model.fit(inputs_n, target)


# In[13]:


model.predict([[2,1,0]])


# In[14]:


model.predict([[2,1,1]])


# In[ ]:




