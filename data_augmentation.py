from tensorflow.keras.layers import Input
import os
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import albumentations as A
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomFlip, RandomHeight, RandomWidth
import cv2 as cv
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import numpy as np


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

def get_index(j):
    if j<10:
        return '00'
    elif j >=10 and j <100:
        return "0"
    else:
        return ""
def data_augmentation(path_1,path_2,path_3,image_path,mask_path):
    transform = A.Compose([A.HorizontalFlip(p=0.5), A.Rotate(limit=20), A.VerticalFlip(p=0.5), A.Transpose(p=0.5)])
    for j in range(15):
        factor=len(path_2)*j
        for i in range(len(path_2)):
            image=cv.imread(path_1[i])
            mask_1=cv.imread(path_2[i])
            mask_2=cv.imread(path_3[i])
            augmented = transform(image=image,mask=mask_2+mask_1)
            image_light = augmented['image']
            image_light_1 = augmented['mask']
            cv.imwrite(image_path+"image_"+get_index(factor+i)+str(factor+i)+".jpg",image_light)
            cv.imwrite(mask_path+"mask_"+get_index(factor+i)+str(factor+i)+".jpg",image_light_1)

