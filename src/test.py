import os
import re
import sys
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow import debugging
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = '../data/interim/image_data/binary/gray_64/train'
validation_data_dir = '../data/interim/image_data/binary/gray_64/val'
nb_train_samples = 480
nb_validation_samples = 60
epochs = 10000
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

opt = keras.optimizers.RMSprop(learning_rate=0.00005,
                                rho=0.9,
                                momentum=0.0,
                                epsilon=1e-07,
                                centered=False)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale= None) # test rescale = 1
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=None) # test rescale = 1

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


model.save('aws_try_128.hdf5')
model.save_weights('aws_try_128.h5')

del model

# model = VGG16(include_top=False,weights=None)

# ## load the locally saved weights 
# model.load_weights("aws_try_128.h5")

# ## show the deep learning model
# model.summary()

# input_img = model.layers[0].input

# # get the symbolic outputs of each "key" layer (we gave them unique names).
# layer_dict = dict([(layer.name, layer) for layer in model.layers])




# plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)
# display(Image.open('cnn_model.png'))