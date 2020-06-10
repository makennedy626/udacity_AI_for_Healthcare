#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample

from itertools import chain
from random import sample 
import scipy


# In[2]:


d = pd.read_csv('findings_data.csv')


# In[3]:


## Here I'm just going to split up my "Finding Labels" column so that I have one column in my dataframe
# per disease, with a binary flag. This makes EDA a lot easier! 

all_labels = np.unique(list(chain(*d['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        d[c_label] = d['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
d.sample(3)


# In[4]:


len(all_labels)


# I see here that there are 14 unique types of labels found in my dataset

# In[5]:


d[all_labels].sum()/len(d)


# In[6]:


ax = d[all_labels].sum().plot(kind='bar')
ax.set(ylabel = 'Number of Images with Label')


# Above, I see the relative frequencies of each disease in my dataset. It looks like 'No Finding' is the most common occurrence. 'No Finding' can never appear with any other label by definition, so we know that in 57.5% of this dataset, there is no finding in the image. Beyond that, it appears that 'Infiltration' is the most common disease-related label, and it is followed by 'Effusion' and 'Atelectasis.'
# 
# Since 'Infiltration' is the most common, I'm going to now look at how frequently it appears with all of the other diseases: 

# In[7]:


##Since there are many combinations of potential findings, I'm going to look at the 30 most common co-occurrences:
plt.figure(figsize=(16,6))
d[d.Infiltration==1]['Finding Labels'].value_counts()[0:30].plot(kind='bar')


# It looks like Infiltration actually occurs alone for the most part, and that its most-common comorbidities are Atelectasis and Effusion. 
# 
# Let's see if the same is true for another label, we'll try Effusion:

# In[8]:


##Since there are many combinations of potential findings, I'm going to look at the 30 most common co-occurrences:
plt.figure(figsize=(16,6))
d[d.Effusion==1]['Finding Labels'].value_counts()[0:30].plot(kind='bar')


# Same thing! Now let's move on to looking at age & gender: 

# In[9]:


plt.figure(figsize=(10,6))
plt.hist(d['Patient Age'])


# In[10]:


plt.figure(figsize=(10,6))
plt.hist(d[d.Infiltration==1]['Patient Age'])


# In[11]:


plt.figure(figsize=(10,6))
plt.hist(d[d.Effusion==1]['Patient Age'])


# Looks like the distribution of age across the whole population is slightly different than it is specifically for Infiltration and Effusion. Infiltration appears to be more skewed towards younger individuals, and Effusion spans the age range but has a large peak around 55. 

# In[12]:


plt.figure(figsize=(6,6))
d['Patient Gender'].value_counts().plot(kind='bar')


# In[13]:


plt.figure(figsize=(6,6))
d[d.Infiltration ==1]['Patient Gender'].value_counts().plot(kind='bar')


# In[14]:


plt.figure(figsize=(6,6))
d[d.Effusion ==1]['Patient Gender'].value_counts().plot(kind='bar')


# Gender distribution seems to be pretty equal in the whole population as well as with Infiltration, with a slight preference towards females in the Effusion distribution. 

# #### Finally, let's look at if and how age & gender relate to mass size in individuals who have a mass as a finding:

# In[15]:


plt.scatter(d['Patient Age'],d['Mass_Size'])


# In[16]:


mass_sizes = d['Mass_Size'].values
mass_inds = np.where(~np.isnan(mass_sizes))
ages = d.iloc[mass_inds]['Patient Age']
mass_sizes=mass_sizes[mass_inds]
scipy.stats.pearsonr(mass_sizes,ages)


# The above tells us that age and mass size are significantly correlated, with a Pearson's coerrelation coefficient of 0.727

# In[17]:


np.mean(d[d['Patient Gender']== 'M']['Mass_Size'])


# In[18]:


np.mean(d[d['Patient Gender']== 'F']['Mass_Size'])


# In[19]:


scipy.stats.ttest_ind(d[d['Patient Gender']== 'F']['Mass_Size'],d[d['Patient Gender']== 'M']['Mass_Size'],nan_policy='omit')


# The above tells us that there is no statistically significant difference between mass size with gender. 
