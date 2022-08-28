from tensorflow.keras.layers import Input
import os
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomFlip, RandomHeight, RandomWidth
import cv2 as cv
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import numpy as np

# loading data from different folder
def load_dataset(image_data, data_mask):
    path_1 = []
    path_2 = []
    total_images = [img for img in sorted(os.listdir(image_data))]
    for img in total_images:
        image_path = image_data + img
        path_1.append(image_path)

    total_mask = [mask for mask in sorted(os.listdir(data_mask))]
    for right in total_mask:
        mask_path = data_mask + right
        path_2.append(mask_path)
    return np.array(path_1), np.array(path_2)


# read the images and masks , resizing to (128,128,3) and (128,128,1),scaling to range [0,1]
def image_preprocessing_1(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (128, 128)) / 255.0
    img = tf.cast(tf.reshape(img, (128, 128, 3)), dtype=tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    total_mask = tf.image.resize(mask, (128, 128)) / 255.
    total_mask = total_mask > 0.6
    total_mask = tf.cast(total_mask, dtype=tf.float32)
    total_mask = tf.reshape(total_mask, (128, 128, 1))
    return img, total_mask


def image_preprocessing_2(image, mask_path, v_1=0.1, v_2=0.9):
    img = tf.io.read_file(image)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (128, 128)) / 255.0
    img = tf.cast(tf.reshape(img, (128, 128, 3)), dtype=tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    total_mask = tf.image.resize(mask, (128, 128)) / 255.
    total_mask = total_mask > 0.6
    total_mask = tf.cast(total_mask, dtype=tf.float32)
    total_mask = tf.reshape(total_mask, (128, 128, 1))
    w_ij = (total_mask * (v_2 - v_1) + v_1)
    mask = (2. * total_mask - 1.) * w_ij
    return img, mask


# data distribution
def train_data_generator(images, mask, model):
    train_dataset = tf.data.Dataset.from_tensor_slices((images, mask))
    train_dataset = train_dataset.map(model).shuffle(32).batch(32).prefetch(4)
    return train_dataset


def valid_data_generator(images, mask, model):
    train_dataset = tf.data.Dataset.from_tensor_slices((images, mask))
    train_dataset = train_dataset.map(model).shuffle(32).batch(32)
    return train_dataset
