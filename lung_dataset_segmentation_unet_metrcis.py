import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


def intersection_over_union(y_true,y_pred):
    axis=range(len(y_true.shape)-1)
    intersection=tf.reduce_sum(y_true*y_pred,axis=axis)
    union = tf.reduce_sum(y_true,axis=axis) + tf.reduce_sum(y_pred,axis=axis) - intersection
    iou=(intersection + 1e-15) / (union + 1e-15)
    return iou
def binary_focal_loss(y_true,y_pred):
    #2
    beta = 0.25
    gamma = 2.
    f_loss = -1*beta*tf.pow((1 - y_pred), gamma) * y_true * tf.math.log(y_pred) - (1-beta)*tf.pow(y_pred, gamma) * (1 - y_true) * tf.math.log(1 - y_pred)
    # β*(1-p̂)ᵞ*p*log(p̂)
    # (1-β)*p̂ᵞ*(1−p)*log(1−p̂)
    # −[β*(1-p̂)ᵞ*p*log(p̂) + (1-β)*p̂ᵞ*(1−p)*log(1−p̂)]
    f_loss=tf.reduce_sum(f_loss)
    return tf.reduce_mean(f_loss)
def dice_coef(y_true,y_pred):
    intersection = tf.reduce_sum(y_true * y_pred,axis=(1,2))
    return tf.reduce_mean((2. * intersection + 1e-15) / (tf.reduce_sum(y_true,axis=(1,2)) + tf.reduce_sum(y_pred,axis=(1,2)) + 1e-15))
def dice_loss(y_true,y_pred):
    return 1.0 - dice_coef(y_true,y_pred)
def soft_dice_loss(y_true,y_pred):
    #3
    part_1=2.*tf.reduce_sum(y_true*y_pred,axis=(1,2))+1e-15
    part_2=tf.reduce_sum(tf.square(y_true),axis=(1,2))+tf.reduce_sum(tf.square(y_pred),axis=(1,2))+1e-15
    soft_loss=tf.reduce_mean(part_1/part_2)
    return 1.-soft_loss
def binary_weighted_dice_cross_entropy_loss(y_true,y_pred):
    #4
    weight=0.7
    loss = weight*dice_loss(y_true,y_pred)+(1-weight*binary_focal_loss(y_true,y_pred))
    return loss
def binary_cross_entropy(y_true,y_pred):
    #1
    bc=BinaryCrossentropy()
    loss=bc(y_true=y_true,y_pred=y_pred)
    return loss
def tversky_index(y_true,y_pred):
    true_pos = tf.reduce_sum(y_true * y_pred,axis=(1,2))
    false_neg = tf.reduce_sum(y_true * (1 - y_pred),axis=(1,2))
    false_pos = tf.reduce_sum((1 - y_true) * y_pred,axis=(1,2))
    alpha = 0.7
    return tf.reduce_mean((true_pos + 1e-15) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + 1e-15))
def tversky_loss(y_true,y_pred):
    #5
    return 1 - tversky_index(y_true,y_pred)


def confusion_matrix(y_true,y_pred):
    y=tf.where(y_pred>0.7,1.0,0.0)
    y_pred_pos = tf.clip_by_value(y, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = tf.clip_by_value(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = tf.reduce_sum(y_pos * y_pred_pos)
    fp = tf.reduce_sum(y_neg * y_pred_pos)
    fn = tf.reduce_sum(y_pos * y_pred_neg)
    precision = (tp + 1e-15) / (tp + fp + 1e-15)
    recall = (tp + 1e-15) / (tp + fn + 1e-15)
    return precision,recall
def focal_tversky(y_true,y_pred):
    #6
    pt_1 = tversky_index(y_true,y_pred)
    gamma = 0.75
    return tf.pow((1. - pt_1), gamma)




