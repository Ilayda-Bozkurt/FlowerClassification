# -*- coding: utf-8 -*-
"""dataloaderVGG16.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jArbl9T3tJb3WE__opvY1WGMzp27_xFM
"""

# Import necessary libraries
import os
import opendatasets as od
import tensorflow as tf
from tensorflow.keras import models,applications,layers,losses,callbacks
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import seaborn as sn
import pathlib
import PIL
import cv2
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from google.colab.patches import cv2_imshow
from tensorflow.keras import applications

#Defining the dataset URL
dataset_url = "https://www.kaggle.com/datasets/alxmamaev/flowers-recognition?datasetId=8782&sortBy=voteCount"

#Defining the data directory
data_dir = './flowers-recognition/flowers'

#If there is no folder, download the dataset
if not os.path.isdir(data_dir):
    od.download(dataset_url)

#List classes
classes = os.listdir(data_dir)
print(classes)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Creating a DataGenerator for training data.The goal: to reduce overfitting.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

#Creating a DataGenerator for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

#Loading training data from folder
train_generator = train_datagen.flow_from_directory(
    'flowers-recognition/flowers',
    target_size=(224, 224),
    batch_size=16,
    class_mode='sparse',
    shuffle=True
)

#Loading validation data from Folder
val_generator = val_datagen.flow_from_directory(
    'flowers-recognition/flowers',
    target_size=(224, 224),
    batch_size=16,
    class_mode='sparse',
    shuffle=False
)

#Define and Train the Model
model = my_model()

#Fitting (Training) the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20
)