
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Importing
x_train = np.load('/content/drive/My Drive/Compition/xs.npy')
y_train = np.load('/content/drive/My Drive/Compition/ys.npy')

def build_model():
  m = Sequential()
  m.add(Conv2D(filters=100, kernel_size=5, input_shape=(32,32,3), activation='relu'))
  m.add(MaxPool2D(pool_size=2, strides=1))
  m.add(BatchNormalization())
  m.add(Conv2D(filters=150, kernel_size=3, activation='relu'))
  m.add(MaxPool2D(pool_size=2, strides=2))
  m.add(BatchNormalization())
  m.add(Flatten())
  m.add(Dense(200,activation='relu'))
  m.add(BatchNormalization())
  m.add(Dropout(0.3))
  m.add(Dense(100,activation='relu'))
  m.add(BatchNormalization())
  m.add(Dropout(0.3))
  m.add(Dense(9,activation='softmax'))
  return m

model = build_model()

early_stopping = EarlyStopping(monitor = 'val_accuracy', patience =  10, restore_best_weights = True)
lr_sechudle = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.2, patience =  6 )

model.compile(optimizer=Adam(1e-5),
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
          epochs = 30,
          verbose =  2,
          callbacks=[early_stopping, lr_sechudle])

model.save('model.h5')