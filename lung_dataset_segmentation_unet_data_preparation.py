
from tensorflow.keras.layers import Input
import os
from lung_dataset_segmentation_unet_model import u_net_segmentation
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation,RandomFlip,RandomHeight,RandomWidth
import cv2 as cv
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import numpy as np





def load_data(image_data,left_data_mask,right_data_mask):
    path_1=[]
    path_2=[]
    path_3=[]

    total_images=[img for img in sorted(os.listdir(image_data))]
    for img in total_images:
        image_path=image_data+img
        path_1.append(image_path)
    total_mask_left=[mask for mask in sorted(os.listdir(left_data_mask))]
    for left in total_mask_left:
        left_path=left_data_mask+left
        path_2.append(left_path)
    total_mask_right=[mask for mask in sorted(os.listdir(right_data_mask))]
    for right in total_mask_right:
        right_path=right_data_mask+right
        path_3.append(right_path)
    return np.array(path_1),np.array(path_2),np.array(path_3)
def image_preprocessing(image,left,right):
    img=tf.io.read_file(image)
    img=tf.image.decode_png(img,channels=3)
    img=tf.image.resize(img,(128,128))/255.0
    img=tf.cast(tf.reshape(img,(128,128,3)),dtype=tf.float32)

    left_one=tf.io.read_file(left)
    left_one=tf.image.decode_png(left_one,channels=1)
    right_one = tf.io.read_file(right)
    right_one = tf.image.decode_png(right_one, channels=1)

    total_mask=left_one+right_one

    total_mask=tf.image.resize(total_mask,(128,128))/255.
    total_mask=total_mask>0.6
    total_mask=tf.cast(total_mask,dtype=tf.float32)
    total_mask=tf.reshape(total_mask,(128,128,1))
    return img,total_mask
def train_data_generator(images,masks_left,masks_right):
    train_dataset=tf.data.Dataset.from_tensor_slices((images,masks_left,masks_right))
    train_dataset=train_dataset.map(image_preprocessing).shuffle(32).batch(32).repeat().prefetch(4)
    return train_dataset
def valid_data_generator(images,masks_left,masks_right):
    train_dataset=tf.data.Dataset.from_tensor_slices((images,masks_left,masks_right))
    train_dataset=train_dataset.map(image_preprocessing).shuffle(32).batch(32).repeat()
    return train_dataset





"""image_data="C://Users/Houssem/Downloads/NLM-MontgomeryCXRSet/MontgomerySet/CXR_png/"
left_data_mask="C://Users/Houssem/Downloads/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/leftMask/"
right_data_mask="C://Users/Houssem/Downloads/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/rightMask/"""
"""images,masks_left,masks_right=load_data(image_data,left_data_mask,right_data_mask)
x_train,x_test=images[:100],images[100:]
y_train_left,y_test_left=masks_left[:100],masks_left[100:]
y_train_right,y_test_right=masks_right[:100],masks_right[100:]"""


"""image_dataset = "C://Users/Houssem/Downloads/Dataset_BUSI_with_GT_14/image_dataset/"
mask_dataset = "C://Users/Houssem/Downloads/Dataset_BUSI_with_GT_14/label_dataset/"""

"""def load_data(image_data,data_mask):
    path_1=[]
    path_2=[]
    total_images=[img for img in os.listdir(image_data)]
    for img in total_images:
        image_path=image_data+img
        path_1.append(image_path)
    total_mask_left=[mask for mask in os.listdir(data_mask)]
    for left in total_mask_left:
        left_path=data_mask+left
        path_2.append(left_path)
    return np.array(path_1),np.array(path_2)"""


"""def image_preprocessing(image,left):
    img=tf.io.read_file(image)
    img=tf.image.decode_png(img,channels=3)
    img=tf.image.resize(img,(128,128))/255.0
    img=tf.cast(tf.reshape(img,(128,128,3)),dtype=tf.float32)
    left_one=tf.io.read_file(left)
    left_one=tf.image.decode_png(left_one,channels=1)
    total_mask = tf.image.resize(left_one, (128, 128)) / 255.
    total_mask = total_mask > 0.6
    total_mask = tf.cast(total_mask, dtype=tf.float32)
    total_mask = tf.reshape(total_mask, (128, 128, 1))
    return np.array(img),np.array(total_mask)

def data_augmentation(image,mask):
    img = image
    annotation = mask
    transform = A.Compose([
        A.HorizontalFlip(p=0.6),
        A.VerticalFlip(p=0.6),
        A.RandomRotate90(p=0.6),
        A.Sharpen(p=0.6),
        A.RandomBrightnessContrast(p=0.6)])
    transformed = transform(image=img,mask=annotation)
    transformed_image,transformed_mask = transformed["image"], transformed["mask"]
    return transformed_image,transformed_mask

def data_train_preparation(image_path,mask_path):
    image, mask = image_preprocessing(image_path,mask_path)
    transformed_image,transformed_mask=data_augmentation(image, mask)
    return transformed_image,transformed_mask
def train_data_generator(images,masks):
    train_dataset=tf.data.Dataset.from_tensor_slices((images,masks))
    train_dataset=train_dataset.map(data_train_preparation).shuffle(32).batch(32).repeat(5).prefetch(4)
    return train_dataset

def valid_data_generator(images,masks):
    train_dataset=tf.data.Dataset.from_tensor_slices((images,masks))
    train_dataset=train_dataset.map(image_preprocessing).shuffle(32).batch(32).repeat()
    return train_dataset"""


"""train_dataset=train_data_generator(x_train,y_train_left,y_train_right)
valid_dataset=valid_data_generator(x_test,y_test_left,y_test_right)"""