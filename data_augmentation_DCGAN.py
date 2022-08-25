import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Conv2D,Reshape,BatchNormalization,Input,Conv2DTranspose,Dense,Activation,LeakyReLU,Dropout,Flatten
from keras.models import Sequential
import os
from tensorflow.keras.optimizers import Adam
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
def generator():
    node=8*8*256
    generator=Sequential()
    generator.add(Dense(node,input_shape=(100,)))
    generator.add(Reshape((8,8,256)))
    generator.add(LeakyReLU(alpha=(0.02)))
    generator.add(Conv2DTranspose(256,strides=2,kernel_size=4,padding="same"))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.02))
    generator.add(Conv2DTranspose(512, strides=2, kernel_size=4, padding="same"))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.02))
    generator.add(Conv2DTranspose(128, strides=2, kernel_size=4, padding="same"))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.02))
    generator.add(Conv2DTranspose(64, strides=2, kernel_size=4, padding="same"))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(alpha=0.02))
    generator.add(Conv2D(4,kernel_size=3,kernel_initializer="he_normal",padding="same"))
    generator.add(BatchNormalization())
    generator.add(Activation("tanh"))
    return generator
def discriminator(image_shape):
    discriminator = Sequential()
    discriminator.add(Input(shape=(128,128,4)))
    discriminator.add(Conv2D(32, kernel_size=3, strides=2, padding="same", input_shape=image_shape))
    discriminator.add(BatchNormalization(axis=-1))
    discriminator.add(LeakyReLU(alpha=0.01))
    discriminator.add(Dropout(0.3))

    discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    discriminator.add(BatchNormalization(axis=-1))
    discriminator.add(LeakyReLU(alpha=0.01))
    discriminator.add(Dropout(0.3))

    discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    discriminator.add(BatchNormalization(axis=-1))
    discriminator.add(LeakyReLU(alpha=0.01))
    discriminator.add(Dropout(0.3))

    discriminator.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    discriminator.add(BatchNormalization(axis=-1))
    discriminator.add(LeakyReLU(alpha=0.01))
    discriminator.add(Dropout(0.3))

    discriminator.add(Conv2D(512, kernel_size=2, strides=2, padding="same"))
    discriminator.add(BatchNormalization(axis=-1))
    discriminator.add(LeakyReLU(alpha=0.01))
    discriminator.add(Dropout(0.3))

    discriminator.add(Conv2D(64, kernel_size=1, strides=1,activation="relu"))

    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation="sigmoid"))
    return discriminator
image_data="C://Users/Houssem/Downloads/NLM-MontgomeryCXRSet/MontgomerySet/CXR_png/"
left_data_mask="C://Users/Houssem/Downloads/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/leftMask/"
right_data_mask="C://Users/Houssem/Downloads/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/rightMask/"

path_1,path_2,path_3=load_data(image_data,left_data_mask,right_data_mask)
print(len(path_1))
def data_preparation(img,left,right):
    image,annotation=image_preprocessing(img,left,right)
    labels=tf.concat([image,annotation],axis=-1)
    return labels



def train_dataset(path_1,path_2,path_3):
    real_dataset=tf.data.Dataset.from_tensor_slices((path_1,path_2,path_3))
    real_dataset=real_dataset.map(data_preparation).batch(32).prefetch(4)
    return real_dataset


model=generator()
discriminator=discriminator((128,128,4))
discriminator.compile(loss="binary_crossentropy",optimizer=Adam(learning_rate=0.001,beta_1=0.5),metrics=["accuracy"])
gan=Sequential([model,discriminator])
gan.compile(loss="binary_crossentropy",optimizer=Adam(learning_rate=0.001),metrics=["accuracy"])

def train_gan(n_epochs,gan,generator,discriminator,random_normal_vector,real_dataset):
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        for real_images in real_dataset:
            batch_size = real_images.shape[0]
            noise_vector = tf.random.normal(shape=[batch_size, random_normal_vector])
            fake_images = generator(noise_vector)
            mixed_image = tf.concat([fake_images, real_images], axis=0)
            discriminator_labels = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(mixed_image, discriminator_labels)
            noise_vector = tf.random.normal(shape=(batch_size, random_normal_vector))
            gan_labels = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise_vector, gan_labels)



