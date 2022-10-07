## 🦉 Lung-Segmentation_in_TensorFlow-2.0
This repository contains the code for the lung segmentation on the Montgomery dataset using UNET and attention UNET architecture using Convolutional Block Attention Module(cbam) in TensorFlow 2.0 framework.

The aim of this work was to compaire between two architecture and find which one can have better performonce by introducing cbam into unet and imporved dice loss by mining the information of negative areas. 

The improved dice loss called weighted dice loss (W_Dice loss). this loss function gives a small weight to the background area of the label, so the
background area will be added to the calculation when calculating dice loss , it's same idea as label smoothing so it can ensure that dice loss is
used to address the unbalanced sample distribution problem and it can deeply mine the information in the positive and negative samples.
![Screenshot 2022-10-07 103039](https://user-images.githubusercontent.com/112108580/194523771-3fd3cfba-7e13-40cf-8521-eb7d92ca16f0.png)





