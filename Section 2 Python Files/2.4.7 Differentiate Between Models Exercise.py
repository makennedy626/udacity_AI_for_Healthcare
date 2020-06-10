#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

import skimage
from skimage import io
import glob

import sklearn
from scipy.ndimage import gaussian_filter


# ## First we'll do background segmentation:

# In[2]:


## Read in two mammo images: 
dense = io.imread('dense/mdb003.pgm')
fatty = io.imread('fatty/mdb005.pgm')


# In[3]:


plt.imshow(dense)


# In[4]:


plt.imshow(fatty)


# In[5]:


x = plt.hist(dense.ravel(),bins=256)


# In[6]:


x = plt.hist(fatty.ravel(),bins=256)


# In[24]:


## Next, experiment with different cut-off intensity thresholds to try to separate the background of the image
## Uncomment the code below and play with the value of 'thresh' to create two new binarized images


thresh = 45

dense_bin = (dense > thresh) * 255
fatty_bin = (fatty > thresh) * 255


# In[25]:


## Visualize the binarized images to see if the threshold you chose separates the breast tissue from the background
plt.imshow(dense_bin)


# In[26]:


plt.imshow(fatty_bin)


# Experiment with different values of 'thresh' above until you are satisfied that you are able to create a reasonable separation of tissue from background.
# 
# One image pre-processing trick you might try before binarizing is _smoothing_ which you perform with a gaussian filter. Try adding the following step before binarization: 
# 
# img_smooth = gaussian_filter(img, sigma = 5)
# 
# Where changing the value of _sigma_ will change the amount of smoothing. 

# ## Once you have chosen your value of thresh, let's use it to see if we can classify dense v. fatty breast tissue: 

# In[27]:


## Let's first get all of the intensity values of the breast tissue for our fatty breast images using the
## segmentation method above
fatty_imgs = glob.glob("fatty/*")
dense_imgs = glob.glob("dense/*")

fatty_intensities = []

for i in fatty_imgs: 
    
    img = plt.imread(i)
    img_mask = (img > thresh)
    fatty_intensities.extend(img[img_mask].tolist())
    
x = plt.hist(fatty_intensities,bins=256)


# In[28]:


## Same for dense breast images 
dense_intensities = []

for i in dense_imgs: 
    
    img = plt.imread(i)
    img_mask = (img > thresh)
    dense_intensities.extend(img[img_mask].tolist())
    
x = plt.hist(dense_intensities,bins=256)


# In[29]:


## Use scipy.stats.mode to get the mode of the two distributions above

dense_mode = scipy.stats.mode(dense_intensities)[0][0]
fatty_mode = scipy.stats.mode(fatty_intensities)[0][0]


# In[30]:


## Loop through all of the fatty images, binarize them using your threshold, and compare the peaks of the 
## distributions of the *tissue only* to the peaks of the distributions of all fatty and all dense breast images: 

for i in fatty_imgs: 
    
    img = plt.imread(i)
    img_mask = (img > thresh)
    
    ## Use scipy.stats.mode to get the mode of the tissue in the image: 
    img_mode = scipy.stats.mode(img[img_mask])[0][0]
    
    fatty_delta = img_mode - fatty_mode
    dense_delta = img_mode - dense_mode
    
    if (np.abs(fatty_delta) < np.abs(dense_delta)):
        print("Fatty")
    else:
        print("Dense")


# In[31]:


## Loop through all of the dense images, binarize them using your threshold, and compare the peaks of the 
## distributions of the *tissue only* to the peaks of the distributions of all fatty and all dense breast images: 

for i in dense_imgs: 
    
    img = plt.imread(i)
    img_mask = (img > thresh)
    
    ## Use scipy.stats.mode to get the mode of the tissue in the image: 
    img_mode = scipy.stats.mode(img[img_mask])[0][0]
    
    fatty_delta = img_mode - scipy.stats.mode(fatty_intensities)[0][0]
    dense_delta = img_mode - scipy.stats.mode(dense_intensities)[0][0]
    
    if (np.abs(fatty_delta) < np.abs(dense_delta)):
        print("Fatty")
    else:
        print("Dense")


# In[ ]:





# In[ ]:




