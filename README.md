## 🦉 Lung-Segmentation_in_TensorFlow-2.0
This repository contains the code for the lung segmentation on the Montgomery dataset using UNET and attention UNET architecture using Convolutional Block Attention Module(cbam) in TensorFlow 2.0 framework.

The aim of this work was to compaire between two architecture and find which one can have better performonce by introducing cbam into unet and imporved dice loss by mining the information of negative areas. 

The improved dice loss called weighted dice loss (W_Dice loss). this loss function gives a small weight to the background area of the label, so the
background area will be added to the calculation when calculating dice loss , it's same idea as label smoothing.




