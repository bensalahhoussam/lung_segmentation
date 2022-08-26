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

image_data = "C://Users/Houssem/Downloads/NLM-MontgomeryCXRSet/MontgomerySet/CXR_png/"
left_data_mask = "C://Users/Houssem/Downloads/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/leftMask/"
right_data_mask = "C://Users/Houssem/Downloads/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/rightMask/"


# loading data from different folder
def load_data(image_data, left_data_mask, right_data_mask):
    path_1 = []
    path_2 = []
    path_3 = []

    total_images = [img for img in sorted(os.listdir(image_data))]
    for img in total_images:
        image_path = image_data + img
        path_1.append(image_path)
    total_mask_left = [mask for mask in sorted(os.listdir(left_data_mask))]
    for left in total_mask_left:
        left_path = left_data_mask + left
        path_2.append(left_path)
    total_mask_right = [mask for mask in sorted(os.listdir(right_data_mask))]
    for right in total_mask_right:
        right_path = right_data_mask + right
        path_3.append(right_path)
    return np.array(path_1), np.array(path_2), np.array(path_3)


# read the images and masks , resizing to (128,128,3) and (128,128,1),scaling to range [0,1]
def image_preprocessing_1(image, left, right, ):
    img = tf.io.read_file(image)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (128, 128)) / 255.0
    img = tf.cast(tf.reshape(img, (128, 128, 3)), dtype=tf.float32)

    left_one = tf.io.read_file(left)
    left_one = tf.image.decode_png(left_one, channels=1)
    right_one = tf.io.read_file(right)
    right_one = tf.image.decode_png(right_one, channels=1)

    total_mask = left_one + right_one

    total_mask = tf.image.resize(total_mask, (128, 128)) / 255.
    total_mask = total_mask > 0.6
    total_mask = tf.cast(total_mask, dtype=tf.float32)
    total_mask = tf.reshape(total_mask, (128, 128, 1))
    return img, total_mask


def image_preprocessing_2(image, left, right, v_1=0.1, v_2=0.9):
    img = tf.io.read_file(image)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (128, 128)) / 255.0
    img = tf.cast(tf.reshape(img, (128, 128, 3)), dtype=tf.float32)

    left_one = tf.io.read_file(left)
    left_one = tf.image.decode_png(left_one, channels=1)
    right_one = tf.io.read_file(right)
    right_one = tf.image.decode_png(right_one, channels=1)

    total_mask = left_one + right_one

    total_mask = tf.image.resize(total_mask, (128, 128)) / 255.
    total_mask = total_mask > 0.6
    total_mask = tf.cast(total_mask, dtype=tf.float32)
    total_mask = tf.reshape(total_mask, (128, 128, 1))
    w_ij = (total_mask * (v_2 - v_1) + v_1)
    mask = (2. * total_mask - 1.) * w_ij
    return img, mask


# data distribution
def train_data_generator(images, masks_left, masks_right, model):
    train_dataset = tf.data.Dataset.from_tensor_slices((images, masks_left, masks_right))
    train_dataset = train_dataset.map(model).shuffle(32).batch(32).prefetch(4)
    return train_dataset


def valid_data_generator(images, masks_left, masks_right, model):
    train_dataset = tf.data.Dataset.from_tensor_slices((images, masks_left, masks_right))
    train_dataset = train_dataset.map(model).shuffle(32).batch(32)
    return train_dataset


path_1, path_2, path_3 = load_data(image_data, left_data_mask, right_data_mask)

train_dataset = train_data_generator(path_1, path_2, path_3)
