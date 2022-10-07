## 🦉 Lung-Segmentation_in_TensorFlow-2.0
This repository contains the code for the lung segmentation on the Montgomery dataset using UNET and attention UNET architecture using Convolutional Block Attention Module(cbam) in TensorFlow 2.0 framework.

The aim of this work was to compaire between two architecture and find which one can have better performonce by introducing cbam into unet and imporved dice loss by mining the information of negative areas. 

The improved dice loss called weighted dice loss (W_Dice loss). this loss function gives a small weight to the background area of the label, so the
background area will be added to the calculation when calculating dice loss , it's same idea as label smoothing so it can ensure that dice loss is
used to address the unbalanced sample distribution problem and it can deeply mine the information in the positive and negative samples pixels by softens 
the one hot type hard label, which reduces the confidence of the positive samples in the label and increases the confidence of the negative samples.

![Screenshot 2022-10-07 103039](https://user-images.githubusercontent.com/112108580/194523771-3fd3cfba-7e13-40cf-8521-eb7d92ca16f0.png)


Attention modules are used to make CNN learn and focus more on the important information, rather than learning non-useful background information. In the case of object classification, useful information is the target class crop that we want to classify and localize in an image.
The attention module consists of a simple 2D-convolutional layer, MLP(in the case of channel attention), and sigmoid function at the end to generate a mask of the input feature map.

![0_DGvAEv6WuMBHT8n8](https://user-images.githubusercontent.com/112108580/194531356-298e3b5e-0616-4342-b517-bea577d36281.png)


## UNET_cabm Architecture

![Screenshot 2022-10-07 092923](https://user-images.githubusercontent.com/112108580/194531952-37bc9242-1a39-4de3-841f-d3a3bfe08be6.jpg)


## Results 

1-Input image
2-Ground truth
3-Predicted mask

![4](https://user-images.githubusercontent.com/112108580/194534051-57a6fcaa-dc32-479f-939f-c857629ca28e.png)
