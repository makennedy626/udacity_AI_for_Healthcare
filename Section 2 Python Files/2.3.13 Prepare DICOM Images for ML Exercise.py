#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pydicom
import glob


# In[2]:


## First, read all of my DICOM files into a list
mydicoms = glob.glob("*.dcm")


# ### Let's look at the contents of the first DICOM:

# In[3]:


dcm1 = pydicom.dcmread(mydicoms[0])
dcm1


# In[6]:


## Do some exploratory work before about how to extract these attributes using pydicom... 
type(dcm1)
print(dir(dcm1))
print(dcm1.PatientID)




# ## Now, let's create the dataframe that we want, and populate it in a loop with all of our DICOMS:
# 
# To complete this exercise, create a single dataframe that has the following columns:
# - Patient ID
# - Patient Age (as an integer)
# - Patient Sex (M/F)
# - Imaging Modality
# - Type of finding in the image
# - Number of rows in the image
# - Number of columns in the image
# 
# Save this dataframe as a .CSV file.

# In[7]:


data_list = []
for dcm in mydicoms:
    mydcm = pydicom.dcmread(dcm)
    columns = [mydcm.PatientID,int(mydcm.PatientAge),mydcm.PatientSex,mydcm.Modality,mydcm.PhotometricInterpretation,mydcm.Rows,mydcm.Columns]
    data_list.append(columns)
    
column_names = ['PatientID','PatientAge','PatientSex','Modality','PhotometricInterpretation','Rows','Columns']    
df = pd.DataFrame(data_list, columns=column_names)
df

