import matplotlib.pyplot as plt
import tensorflow as tf
from data_preprocessing import image_preprocessing_1,image_preprocessing_2,load_data
from u_net_train import model_1,model_2
from u_net_train import x_test,y_test_left,y_train_right


def prediction(image_path,image_left_path,image_right_path,model,preparation):
    image,mask=preparation(image_path,image_left_path,image_right_path)
    image=tf.expand_dims(image,axis=0)
    y_output=model (image)
    mask=tf.concat([mask,mask,mask],axis=-1)
    y_output=tf.concat([y_output,y_output,y_output],axis=-1)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.subplot(1, 3, 3)
    plt.imshow(y_output)
    plt.show()


prediction(x_test[16], y_test_left[16], y_train_right[16], model_1, image_preprocessing_1)