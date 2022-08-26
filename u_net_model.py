import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Dropout,Dense,concatenate,BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def conv_block(input, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding="same",kernel_initializer="he_normal")(input)
    x=BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2D(num_filters, kernel_size=3, padding="same",kernel_initializer="he_normal")(x)
    x=BatchNormalization()(x)
    x = Activation("elu")(x)
    return x
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D(pool_size=(2,2))(x)
    p = Dropout(0.3)(p)
    return x, p
# encoder block
def encoder(inputs):
    f1, p1 = encoder_block(inputs, num_filters=64)
    f2, p2 = encoder_block(p1, num_filters=128)
    f3, p3 = encoder_block(p2, num_filters=256)
    f4, p4 = encoder_block(p3, num_filters=512)
    return p4, (f1, f2, f3, f4)
# bottleneck block
def bottleneck(inputs):
    bottle_neck = conv_block(inputs, num_filters=1024)
    return bottle_neck

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (3,3), strides=2, padding="same",kernel_initializer="he_normal",use_bias=False)(input)
    x = Concatenate()([x, skip_features])
    x = Dropout(0.3)(x)
    x = conv_block(x, num_filters)
    return x
# decoder block
def decoder(inputs, convs):
    f1, f2, f3, f4 = convs
    c6 = decoder_block(inputs, f4, num_filters=512)
    c7 = decoder_block(c6, f3, num_filters=256)
    c8 = decoder_block(c7, f2, num_filters=128)
    c9 = decoder_block(c8, f1, num_filters=64)
    outputs = Conv2D(1, 1, kernel_initializer="he_normal",activation='sigmoid')(c9)
    return outputs

def u_net():
    inputs = Input(shape=(128, 128, 3,))
    encoder_output, convs = encoder(inputs)
    bottle_neck = bottleneck(encoder_output)
    outputs = decoder(bottle_neck, convs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile()
    return model


