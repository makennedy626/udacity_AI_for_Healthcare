#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import pandas as pd
import numpy as np
import pydicom
import skimage
import matplotlib.pyplot as plt


# #### First, read in your dataframe of images with bounding box coordinates

# In[2]:


bbox = pd.read_csv('bounding_boxes.csv')
bbox


# #### For each of the three DICOM files listed in the dataframe, do the following: 
# 1. Read the DICOM's pixel_array attribute into a dataframe using the pydicom.dcmread function
# 2. Visualize the image using plt.imshow
# 3. Plot a histogram of the image pixel intensity values
# 4. Find the mean and standard deviation of intensity values of the image, and standardize it using the standardization formula:
# ```test
# (X - X_mean)/X_std_dev
# ```
# 
# 5. Re-plot a histogram of the normalized intensity values
# 6. Use the coordinates in the dataframe that tell the starting x & y values, and the width and height of the mass to plot visualize only the mass using plt.imshow
# 7. Plot a histogram of the normalized intensity values of the mass

# In[3]:


image_index_list = bbox['Image Index'].tolist()
first_dcm = pydicom.dcmread(image_index_list[0])
plt.imshow(first_dcm.pixel_array,cmap='gray')


# In[4]:


plt.hist(first_dcm.pixel_array.ravel(), bins=256)
mean_intensity_1 = np.mean(first_dcm.pixel_array)
std_intensity_1 = np.std(first_dcm.pixel_array)
new_img_1 = first_dcm.pixel_array.copy()
new_img_1 = (new_img_1 - mean_intensity_1)/std_intensity_1


# In[5]:


plt.figure(figsize=(5,5))
plt.hist(new_img_1.ravel(), bins=256)
plt.show()


# In[6]:


image_index_list = bbox['Image Index'].tolist()
x_list = bbox.iloc[:,3].tolist()
y_list = bbox.iloc[:,4].tolist()
w_list = bbox.iloc[:,5].tolist()
h_list = bbox.iloc[:,6].tolist()

my_dicom_1 = pydicom.dcmread(image_index_list[0])
my_dicom_1_x = [x_list[0],x_list[0]+w_list[0]]
my_dicom_1_y = [y_list[0],y_list[0]+h_list[0]]
x_1 = int(x_list[0])
x_2 = int(x_list[0]+w_list[0])
y_1 = int(y_list[0])
y_2 = int(y_list[0]+h_list[0])
# my_dicom_1_image = my_dicom_1.pixel_array
plt.imshow(first_dcm.pixel_array[x_1:x_2,y_1:y_2],cmap='gray')

# plt.hist(my_dicom_1[[my_dicom_1_y[0]:my_dicom_1_y[1],my_dicom_1_x[0]:my_dicom_1_x[1]].ravel(), bins = 256,color='green')

# my_dicom_2 = pydicom.dcmread(image_index_list[1])
# my_dicom_2_x = [x_list[1],x_list[1]+w_list[1]]
# my_dicom_2_y = [y_list[1],y_list[1]+h_list[1]]

# my_dicom_3 = pydicom.dcmread(image_index_list[2])
# my_dicom_3_x = [x_list[2],x_list[2]+w_list[2]]
# my_dicom_3_y = [y_list[2],y_list[2]+h_list[2]]


# In[7]:


plt.hist(new_img_1[x_1:x_2,y_1:y_2].ravel(),bins=256,color='red')
plt.show()

