import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model, Sequential, Model
from data_preprocessing import load_data, train_data_generator, valid_data_generator
from u_net_model import u_net
from metrcis import intersection_over_union, binary_cross_entropy, tversky_loss, focal_tversky, \
    binary_weighted_dice_cross_entropy_loss, dice_coefficient, tversky_index
from graph import plot_history, confusion_matrix
from u_net_Se_ResNextt import u_net_1
from keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

image_data = "C://Users/Houssem/Downloads/NLM-MontgomeryCXRSet/MontgomerySet/CXR_png/"
left_data_mask = "C://Users/Houssem/Downloads/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/leftMask/"
right_data_mask = "C://Users/Houssem/Downloads/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/rightMask/"

path_1, path_2, path_3 = load_data(image_data, left_data_mask, right_data_mask)

x_train, x_test = path_1[:100], path_1[100:]
y_train_left, y_test_left = path_2[:100], path_2[100:]
y_train_right, y_test_right = path_3[:100], path_3[100:]

train_dataset = train_data_generator(x_train, y_train_left, y_train_right)
valid_dataset = valid_data_generator(x_test, y_test_left, y_test_right)

checkpoint = ModelCheckpoint("D://Deep_Learning_projects/new_projects/computer_vision",
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                           restore_best_weights=True)

model_1 = u_net()
model_2 = u_net_1()

model_1.compile(loss=binary_cross_entropy, optimizer=Adam(learning_rate=0.001),
                metrics=[dice_coefficient, tversky_index, intersection_over_union])

history = model_1.fit(train_dataset, steps_per_epoch=len(train_dataset), epochs=100, validation_data=valid_dataset,
                      validation_steps=len(valid_dataset), callbacks=[early_stop, checkpoint])
