#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import io

from keras.preprocessing.image import ImageDataGenerator


# In[2]:


df = pd.read_csv('paths_and_labels.csv')


# In[3]:


df.head()


# In[4]:


df.img_path[0]


# ## First, let's look at what our plan images look like:

# In[5]:


fig, m_axs = plt.subplots(5,4, figsize = (16, 16))
m_axs = m_axs.flatten()
imgs = df.img_path
ind=0

for img, ax in zip(imgs, m_axs):
    img = io.imread(img)
    ax.imshow(img,cmap='gray')
    ax.set_title(df.iloc[ind]['class'])
    ind=ind+1


# In[6]:


## you can choose an image size to resize your images to during augmentation, which comes in handy later when you
## want to do deep learning using a pre-trained CNN that has a specified size input layer
IMG_SIZE = (224, 224)


# ## Below, you will alter the values of the parameters that you pass to ImageDataGenerator. The following cell, you'll display what your augmented images look like. 
# 
# #### Play around with different values for the parameters, running the visualization cell below each time to see how these parameters change the appearance of your augmented data. Make some conclusions about what sorts of values might or might not be appropriate for medical imaging based on what you might see in the real world. You can look at the ImageDataGenerator documentation in Keras to add other parameters as well. 
# 
# * horizontal_flip and vertical_flip should be set to True/False
# * height_shift_range and width_shift_range should be between 0 and 1
# * rotation_range can be between 0 and 180
# * shear_range can be between 0 and 1
# * zoom_range can be between 0 and 1

# In[7]:


idg = ImageDataGenerator(rescale=1. / 255.0,
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.1, 
                              width_shift_range=0.1, 
                              rotation_range=20, 
                              shear_range = 0.1,
                              zoom_range=0.1)

gen = idg.flow_from_dataframe(dataframe=df, 
                                         directory=None, 
                                         x_col = 'img_path',
                                         y_col = 'class',
                                         class_mode = 'binary',
                                         target_size = IMG_SIZE, 
                                         batch_size = 9
                                         )


# In[8]:


## Look at some examples of our augmented training data. 
## This is helpful for understanding the extent to which data is being manipulated prior to training, and can be compared
## With how the raw data look prior to augmentation
t_x, t_y = next(gen)
fig, m_axs = plt.subplots(3, 3, figsize = (8, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    if c_y == 1: 
        c_ax.set_title('fatty')
    else:
        c_ax.set_title('dense')
    c_ax.axis('off')


# These look reasonable from a medical imaging point of view. Let's see what happens when we choose different parameters (I'm just copying the above two cells and running them again below with other parameters.) 

# In[9]:


idg = ImageDataGenerator(rescale=1. / 255.0,
                              horizontal_flip = True, 
                              vertical_flip = True, ## now i'm adding vertical flip
                              height_shift_range= 0.1, 
                              width_shift_range=0.1, 
                              rotation_range=45, ## I'm also increasing the rotation_range
                              shear_range = 0.1,
                              zoom_range=0.1)

gen = idg.flow_from_dataframe(dataframe=df, 
                                         directory=None, 
                                         x_col = 'img_path',
                                         y_col = 'class',
                                         class_mode = 'binary',
                                         target_size = IMG_SIZE, 
                                         batch_size = 9
                                         )


# In[10]:


## Look at some examples of our augmented training data. 
## This is helpful for understanding the extent to which data is being manipulated prior to training, and can be compared
## With how the raw data look prior to augmentation
t_x, t_y = next(gen)
fig, m_axs = plt.subplots(3, 3, figsize = (8, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    if c_y == 1: 
        c_ax.set_title('fatty')
    else:
        c_ax.set_title('dense')
    c_ax.axis('off')


# These do *not* look reasonable. There are upside-down images because I added vertical flip, which we'd never see in a clinical setting. So, we _don't_ want to use this type of augmentation with medical images.

# In[11]:


idg = ImageDataGenerator(rescale=1. / 255.0,
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.1, 
                              width_shift_range=0.1, 
                              rotation_range=20,
                              shear_range = 0.1,
                              zoom_range=0.5) ## Here I'm adding a lot more zoom 

gen = idg.flow_from_dataframe(dataframe=df, 
                                         directory=None, 
                                         x_col = 'img_path',
                                         y_col = 'class',
                                         class_mode = 'binary',
                                         target_size = IMG_SIZE, 
                                         batch_size = 9
                                         )


# In[12]:


## Look at some examples of our augmented training data. 
## This is helpful for understanding the extent to which data is being manipulated prior to training, and can be compared
## With how the raw data look prior to augmentation
t_x, t_y = next(gen)
fig, m_axs = plt.subplots(3, 3, figsize = (8, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    if c_y == 1: 
        c_ax.set_title('fatty')
    else:
        c_ax.set_title('dense')
    c_ax.axis('off')


# These don't look too bad, although it's possible that too much zoom was added. There's no "right" answer for this one, just gaining an understanding of how these parameters change your augmented images. 

# In[ ]:




