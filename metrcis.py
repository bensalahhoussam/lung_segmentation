import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


def intersection_over_union(y_true, y_output):
    axis = range(len(y_true.shape) - 1)
    intersection = tf.reduce_sum(y_true * y_output, axis=axis)
    union = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_output, axis=axis) - intersection
    iou = (intersection + 1e-15) / (union + 1e-15)
    return iou


def binary_focal_loss(y_true, y_output):
    # 2
    beta = 0.25
    gamma = 2.
    # β*(1-p̂)ᵞ*p*log(p̂)
    # (1-β)*p̂ᵞ*(1−p)*log(1−p̂)
    # −[β*(1-p̂)ᵞ*p*log(p̂) + (1-β)*p̂ᵞ*(1−p)*log(1−p̂)]
    f_loss = -1 * beta * tf.pow((1 - y_output), gamma) * y_true * tf.math.log(y_output) - (1 - beta) * \
             tf.pow(y_output, gamma) * (1 - y_true) * tf.math.log(1 - y_output)

    f_loss = tf.reduce_sum(f_loss)
    return tf.reduce_mean(f_loss)


def dice_coefficient(y_true, y_output):
    intersection = tf.reduce_sum(y_true * y_output, axis=(1, 2))
    return tf.reduce_mean(
        (2. * intersection + 1e-15) / (
                tf.reduce_sum(y_true, axis=(1, 2)) + tf.reduce_sum(y_output, axis=(1, 2)) + 1e-15))


def dice_loss(y_true, y_output):
    return 1.0 - dice_coefficient(y_true, y_output)


def soft_dice_loss(y_true, y_output):
    part_1 = 2. * tf.reduce_sum(y_true * y_output, axis=(1, 2)) + 1e-15
    part_2 = tf.reduce_sum(tf.square(y_true), axis=(1, 2)) + tf.reduce_sum(tf.square(y_output), axis=(1, 2)) + 1e-15
    soft_loss = tf.reduce_mean(part_1 / part_2)
    return 1. - soft_loss


def binary_weighted_dice_cross_entropy_loss(y_true, y_output):
    weight = 0.7
    loss = weight * dice_loss(y_true, y_output) + (1 - weight * binary_focal_loss(y_true, y_output))
    return loss


def binary_cross_entropy(y_true, y_output):
    bc = BinaryCrossentropy()
    loss = bc(y_true=y_true, y_pred=y_output)
    return loss


def tversky_index(y_true, y_output):
    true_pos = tf.reduce_sum(y_true * y_output, axis=(1, 2))
    false_neg = tf.reduce_sum(y_true * (1 - y_output), axis=(1, 2))
    false_pos = tf.reduce_sum((1 - y_true) * y_output, axis=(1, 2))
    alpha = 0.7
    return tf.reduce_mean((true_pos + 1e-15) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + 1e-15))


def tversky_loss(y_true, y_output):
    # 5
    return 1 - tversky_index(y_true, y_output)

def focal_tversky(y_true, y_output):
    # 6
    pt_1 = tversky_index(y_true, y_output)
    gamma = 0.75
    return tf.pow((1. - pt_1), gamma)


