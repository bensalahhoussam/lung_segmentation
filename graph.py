import matplotlib.pyplot as plt
import tensorflow as tf

def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def confusion_matrix(y_true, y_output):
    y = tf.where(y_output > 0.7, 1.0, 0.0)
    y_output_pos = tf.clip_by_value(y, 0, 1)
    y_output_neg = 1 - y_output_pos
    y_pos = tf.clip_by_value(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = tf.reduce_sum(y_pos * y_output_pos)
    fp = tf.reduce_sum(y_neg * y_output_pos)
    fn = tf.reduce_sum(y_pos * y_output_neg)
    precision = (tp + 1e-15) / (tp + fp + 1e-15)
    recall = (tp + 1e-15) / (tp + fn + 1e-15)
    return precision, recall