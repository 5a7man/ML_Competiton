# -*- coding: utf-8 -*-
"""TL_vgg16_kerasAPI.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10QTvjCnU8hkrF7gOIaDqy2ncXYCMprmV
"""

import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model

# Data Importing
x_train = np.load('/content/drive/My Drive/Compition/xs.npy')
y_train = np.load('/content/drive/My Drive/Compition/ys.npy')
x_train = x_train/255.0

def build_model():
  vgg16 = VGG16(include_top = False, input_shape=(32,32,3),weights='imagenet')
  for layer in vgg16.layers:
    layer.trainable = False
  x = Flatten()(vgg16.output)
  x = Dense(500,activation='relu')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.3)(x)
  x = Dense(250,activation='relu')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.25)(x)
  x = Dense(70,activation='relu')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)
  x = Dense(9,activation='softmax')(x)
  m = Model(inputs = vgg16.inputs, outputs = x)
  return m

model = build_model()

early_stopping = EarlyStopping(monitor = 'val_accuracy', patience =  20, restore_best_weights = True)
lr_sechudle = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.2, patience =  12 )

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