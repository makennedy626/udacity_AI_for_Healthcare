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
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau


# In[2]:


train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('test.csv')


# ## Setting up the image augmentation from last Lesson: 

# In[3]:


IMG_SIZE = (224, 224)


# In[4]:


train_idg = ImageDataGenerator(rescale=1. / 255.0,
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.1, 
                              width_shift_range=0.1, 
                              rotation_range=20, 
                              shear_range = 0.1,
                              zoom_range=0.1)

train_gen = train_idg.flow_from_dataframe(dataframe=train_df, 
                                         directory=None, 
                                         x_col = 'img_path',
                                         y_col = 'class',
                                         class_mode = 'binary',
                                         target_size = IMG_SIZE, 
                                         batch_size = 9
                                         )

# Note that the validation data should not be augmented! We only want to do some basic intensity rescaling here
val_idg = ImageDataGenerator(rescale=1. / 255.0
                                 )

val_gen = val_idg.flow_from_dataframe(dataframe=valid_df, 
                                         directory=None, 
                                         x_col = 'img_path',
                                         y_col = 'class',
                                         class_mode = 'binary',
                                         target_size = IMG_SIZE, 
                                         batch_size = 6) ## We've only been provided with 6 validation images


# In[5]:


## Pull a single large batch of random validation data for testing after each epoch
testX, testY = val_gen.next()


# ## Now we'll load in VGG16 with pre-trained ImageNet weights: 

# In[6]:


model = VGG16(include_top=True, weights='imagenet')
model.summary()


# In[7]:


transfer_layer = model.get_layer('block5_pool')
vgg_model = Model(inputs=model.input,
                   outputs=transfer_layer.output)


# In[8]:


## Now, choose which layers of VGG16 we actually want to fine-tune (if any)
## Here, we'll freeze all but the last convolutional layer
for layer in vgg_model.layers[0:17]:
    layer.trainable = False


# In[9]:


for layer in vgg_model.layers:
    print(layer.name, layer.trainable)


# In[10]:


new_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
new_model.add(vgg_model)

# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
new_model.add(Flatten())

# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(1, activation='sigmoid'))


# In[11]:


## Set our optimizer, loss function, and learning rate
optimizer = Adam(lr=1e-4)
loss = 'binary_crossentropy'
metrics = ['binary_accuracy']


# In[12]:


new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# In[13]:


## Just run a single epoch to see how it does:
new_model.fit_generator(train_gen, 
                                  validation_data = (testX, testY), 
                                  epochs = 5)


# ## Let's try another experiment where we add a few more dense layers:

# In[14]:


new_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
new_model.add(vgg_model)

# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
new_model.add(Flatten())

# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(1024, activation='relu'))

# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(512, activation='relu'))

# Add a dense (aka. fully-connected) layer.
# Change the activation function to sigmoid 
# so output of the last layer is in the range of [0,1] 
new_model.add(Dense(1, activation='sigmoid'))


# In[15]:


new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# In[16]:


## Just run a single epoch to see how it does:
new_model.fit_generator(train_gen, 
                                  validation_data = (testX, testY), 
                                  epochs = 5)


# ## Now let's add dropout and another fully connected layer:

# In[17]:


new_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
new_model.add(vgg_model)

# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
new_model.add(Flatten())

# Add a dropout-layer which may prevent overfitting and
# improve generalization ability to unseen data e.g. the test-set.
new_model.add(Dropout(0.5))

# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(1024, activation='relu'))

# Add a dropout-layer which may prevent overfitting and
# improve generalization ability to unseen data e.g. the test-set.
new_model.add(Dropout(0.5))

# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(512, activation='relu'))

# Add a dropout-layer which may prevent overfitting and
# improve generalization ability to unseen data e.g. the test-set.
new_model.add(Dropout(0.5))

# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(256, activation='relu'))

# Add a dense (aka. fully-connected) layer.
# Change the activation function to sigmoid 
# so output of the last layer is in the range of [0,1] 
new_model.add(Dense(1, activation='sigmoid'))


# In[18]:


new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# In[19]:


## Just run a single epoch to see how it does:
new_model.fit_generator(train_gen, 
                                  validation_data = (testX, testY), 
                                  epochs = 5)


# What's interesting about the small number of epochs we ran on the three different architectures above is that the simplest archiecture seemed to show the fastest learning. Why might that be? 
# 
# Answer: there were the fewest parameters to train because we didn't add any fully-connected layers, and were only fine-tuning the last layer of VGG16. 
# 
# The last architecture we tried seemed to show more stable and promise than the second, and this is likely due to the fact that we added Dropout. This helps our model from overfitting and usually using Dropout, we see better learning on the validation set (val_loss going down over epochs as opposed to only the training loss). 

# In[ ]:




