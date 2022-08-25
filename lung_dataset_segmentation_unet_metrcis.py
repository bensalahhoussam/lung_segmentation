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


"""image_dataset="C://Users/Houssem/Downloads/Dataset_BUSI_with_GT_14/image_dataset/"
mask_dataset="C://Users/Houssem/Downloads/Dataset_BUSI_with_GT_14/label_dataset/"

path_1,path_2=load_data_1(image_dataset,mask_dataset)
image_1,mask_1=image_preprocessing_1(path_1[0],path_2[0])
image_2,mask_2=image_preprocessing_1(path_1[1],path_2[1])
image_3,mask_3=image_preprocessing_1(path_1[2],path_2[2])
image_4,mask_4=image_preprocessing_1(path_1[3],path_2[3])
image_5,mask_5=image_preprocessing_1(path_1[4],path_2[4])
image_6,mask_6=image_preprocessing_1(path_1[5],path_2[5])
image_7,mask_7=image_preprocessing_1(path_1[6],path_2[6])
image_8,mask_8=image_preprocessing_1(path_1[7],path_2[7])
image=tf.stack([image_1,image_2,image_3,image_4,image_5,image_6,image_7,image_8],axis=0)
mask=tf.stack([mask_1,mask_2,mask_3,mask_4,mask_5,mask_6,mask_7,mask_8],axis=0)
print(mask.shape)
y_pred=u_net_segmentation(image)
print(f"y_pred shape :{y_pred.shape}")
loss=binary_focal_loss(y_true=mask,y_pred=y_pred)
print(loss)
binary_loss=binary_cross_entropy(mask,y_pred)
print(binary_loss)"""
"""image_dataset="C://Users/Houssem/Downloads/Dataset_BUSI_with_GT_14/image_dataset/"
mask_dataset="C://Users/Houssem/Downloads/Dataset_BUSI_with_GT_14/label_dataset/"
path_1,path_2=load_data_1(image_dataset,mask_dataset)
image,mask=image_preprocessing_1(path_1[0],path_2[0])
y_true=tf.expand_dims(mask,axis=0)
image=tf.expand_dims(image,axis=0)
y_pred=u_net_segmentation(image)

loss=bce_dice_loss(y_true,y_pred)
print(loss)"""
"""y_true=np.array([[1.,0.,1.,1.,0.],[0.,1.,1.,0.,0.],[1.,1.,0.,1.,1.]])
y_true=np.reshape(y_true,newshape=(1,3,5,1))
print(y_true.shape)
y_pred=np.array([[0.2,0.8,0.4,0.85,0.15],[0.9,0.2,0.85,0.88,0.25],[0.5,0.12,0.8,0.45,0.35]])
y_pred=np.reshape(y_pred,newshape=(1,3,5,1))
print(y_pred.shape)
loss=binary_crossentropy(y_true=y_true,y_pred=y_pred)
print(f"loss:{np.mean(loss)}")
print(f"loss:{loss.shape}")"""
"""def bce(y_true,y_pred,alpha,gamma):
    loss=-alpha*((1-y_pred)**gamma)*y_true*np.log(y_pred)-(1-alpha)*(y_pred**gamma)*(1-y_true)*np.log(1-y_pred)
    loss=np.mean(loss)
    return loss

loss_1=bce(y_true,y_pred,0.25,2)
print(f"loss_1:{loss_1}")

c=BinaryCrossentropy()
losses=c(y_true,y_pred)
print(losses)

loss_2=binary_cross_entropy(y_true,y_pred)
print(loss_2)
focal=binary_focal_loss(y_true,y_pred)
print(focal)"""



