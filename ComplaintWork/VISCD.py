#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import pickle


# In[7]:


with open("./processed/tokenized_1.csv", "rb") as fd:
    complaints = pickle.load(fd)


# In[8]:


complaints.head()


# In[15]:


types = complaints.Product.unique().tolist()


# In[16]:


types


# In[20]:


complaints["cnt"] = complaints["tokens"].apply(lambda x: len(x))


# In[21]:


complaints["cnt"]


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")


# In[22]:


f, ax = plt.subplots(figsize=(6, 15))
sns.barplot(x="cnt", y="Product", data=complaints,
            label="Total", color="b")


# In[11]:


complaints.iloc[0].tokens[:5]


# In[25]:


complaints = pd.read_csv("./complaints/complaints.csv")


# In[26]:


complaints.head()


# In[27]:


complaints.dtypes


# In[30]:


from tqdm import tqdm
tqdm.pandas()


complaints["Date received"] = complaints["Date received"].progress_apply(pd.to_datetime)


# In[31]:


complaints.dtypes


# In[33]:


complaints["year"] = complaints["Date received"].progress_apply(lambda x: x.year)


# In[34]:


complaints["month"] = complaints["Date received"].progress_apply(lambda x: x.month)


# In[41]:


plt.figure(figsize=(20,20))

sns.countplot(y="year",  hue="Product", data=complaints)


# In[42]:


plt.figure(figsize=(20,20))

sns.countplot(y="month",  hue="Product", data=complaints)

