import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Model

# Data Importing
x_train = np.load('/content/drive/My Drive/Compition/xs.npy')
y_train = np.load('/content/drive/My Drive/Compition/ys.npy')

def build_model():
  resNet50 = ResNet50(include_top = False, input_shape=(32,32,3),weights='imagenet')
  for layer in resNet50.layers:
    layer.trainable = False
  x = Flatten()(resNet50.output)
  x = Dense(200,activation='relu')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.3)(x)
  x = Dense(100,activation='relu')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.3)(x)
  x = Dense(9,activation='softmax')(x)
  m = Model(inputs = resNet50.inputs, outputs = x)
  return m

model = build_model()

early_stopping = EarlyStopping(monitor = 'val_accuracy', patience =  10, restore_best_weights = True)
lr_sechudle = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.2, patience =  6 )

model.compile(optimizer=Adam(1e-4),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

generator = ImageDataGenerator(rotation_range=90,
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