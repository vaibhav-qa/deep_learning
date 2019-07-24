from tensorflow import keras
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
mypath = 'D:\\ml\\test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "D:\\dogs-vs-cats_train", target_size=(150,150), batch_size=20, class_mode='binary')


validation_generator = test_datagen.flow_from_directory(
    "D:\\dogs-vs-cats_validation", target_size=(150,150), batch_size=20,class_mode='binary')

model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(128,(3,3), activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(128,(3,3), activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')])

model.load_weights("D:\\ml\\cats_and_dogs_small_1.h5")

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for image_file in onlyfiles:
    img = image.load_img("D:\\ml\\test\\" + image_file, target_size=(150,150))
    x = image.img_to_array(img)
    x = x.reshape((1,)+x.shape)
    print(model.predict(x))
    if model.predict(x) < 0.3:
        print(image_file + ": Must be a cat")
    if model.predict(x) > 0.7:
        print(image_file + ": Must be a dog")
    if model.predict(x) > 0.3 and model.predict(x) < 0.7:
        print(image_file + ": Not sure if its a cat or a dog")


