# -*- coding: utf-8 -*-
"""LeNET_23_11_2020.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hb5fbap_RjGgzdVi3HyXFxRSNI0O8G57
"""

seed_value=42

import os
os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)

from tensorflow.keras.layers import InputLayer,Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

# Data Importing
x_train = np.load('/content/drive/My Drive/Compition/xs.npy')
y_train = np.load('/content/drive/My Drive/Compition/ys.npy')

def build_model():
  m = Sequential()
  m.add(InputLayer(input_shape=(32,32,3)))
  m.add(Rescaling(scale=1./255,offset=0.0))
  m.add(Conv2D(filters=100, kernel_size=5, activation='relu'))
  m.add(MaxPool2D(pool_size=2, strides=2))
  m.add(BatchNormalization())
  m.add(Conv2D(filters=150, kernel_size=3, activation='relu'))
  m.add(MaxPool2D(pool_size=2, strides=2))
  m.add(BatchNormalization())
  m.add(Conv2D(filters=200, kernel_size=3, activation='relu'))
  m.add(MaxPool2D(pool_size=2, strides=2))
  m.add(BatchNormalization())
  m.add(Flatten())
  m.add(Dense(300,activation='relu'))
  m.add(BatchNormalization())
  m.add(Dropout(0.1))
  m.add(Dense(150,activation='relu'))
  m.add(BatchNormalization())
  m.add(Dropout(0.1))
  m.add(Dense(75,activation='relu'))
  m.add(BatchNormalization())
  m.add(Dropout(0.1))
  m.add(Dense(9,activation='softmax'))
  return m

model = build_model()

early_stopping = EarlyStopping(monitor = 'val_accuracy', patience =  10, restore_best_weights = True)
lr_sechudle = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.2, patience =  6 )

model.compile(optimizer=Adam(1e-3),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

generator = ImageDataGenerator(rotation_range=20,
                               zoom_range = 0.2,
                               horizontal_flip = True,
                               vertical_flip = True,
                               width_shift_range = 0.1,
                               height_shift_range = 0.1,
                               validation_split = 0.2)

model.fit(generator.flow(x_train,y_train,batch_size = 25),
          validation_data = generator.flow(x_train,y_train,batch_size = 25, subset = 'validation'),
          steps_per_epoch = len(x_train)/25,
          epochs = 100,
          verbose =  2,
          callbacks=[early_stopping, lr_sechudle])

model.save('model.h5')