# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 04:59:05 2018

@author: Rishabh Sharma
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_files
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D
from keras.applications import Xception
from keras.models import Model

train_directory = "../../Dermatologist_Dataset/data/train/"
test_directory = "../../Dermatologist_Dataset/data/test/"
validation_directory = "../../Dermatologist_Dataset/data/valid/"

xception = Xception(weights="imagenet", include_top = False, input_shape=(224,224,3))

model = xception.output
model = Flatten()(model)
model = Dense(256, activation='relu', input_dim=7 * 7 * 512)(model)
model = Dropout(0.5)(model)
output_ = Dense(3, activation='softmax')(model)

model = Model(inputs = xception.input, outputs = output_)

model.compile(loss="categorical_crossentropy", optimizer = 'adam', metrics=["accuracy"])


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

train_generator = datagen.flow_from_directory(
    train_directory,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle="True")

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(test_directory,
                                            target_size = (224, 224),
                                            batch_size = batch_size,
                                            shuffle = False,
                                            class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(validation_directory,
                                                        target_size = (224,224),
                                                        batch_size = batch_size,
                                                        shuffle = False,
                                                        class_mode = "categorical")


model.fit_generator(
train_generator,
samples_per_epoch = 1000,
epochs = 5,
validation_data = validation_generator,
nb_val_samples = 100)