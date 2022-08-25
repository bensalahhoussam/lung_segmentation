import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D,Input,Conv2D,MaxPooling2D,GlobalMaxPooling2D,Reshape,Dense,Add,Activation,Multiply,Lambda,Concatenate,Conv2DTranspose,BatchNormalization
from tensorflow.keras.models import Model,load_model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model

def perception_layer(input_feature,ratio):
    shared_layer_one=Dense(input_feature.shape[-1]//ratio,kernel_initializer="he_normal",activation="relu")(input_feature)
    shared_layer_two=Dense(input_feature.shape[-1],kernel_initializer="he_normal",activation="sigmoid")(shared_layer_one)
    return shared_layer_two
def channel_attention(input_feature):
    average_pooling=GlobalAveragePooling2D()(input_feature)
    average_pooling=Reshape((1,1,input_feature.shape[-1]))(average_pooling)

    max_pooling=GlobalMaxPooling2D()(input_feature)
    max_pooling=Reshape((1,1,input_feature.shape[-1]))(max_pooling)

    avg_output=perception_layer(average_pooling,8)
    max_output=perception_layer(max_pooling,8)

    attention=Add()([avg_output,max_output])
    attention=Activation("sigmoid")(attention)
    attention=Multiply()([input_feature,attention])
    return attention
def spatial_attention(input_feature):
    avg_pool=Lambda(lambda x:K.mean(x,axis=-1,keepdims=True))(input_feature)
    max_pool=Lambda(lambda x:K.max(x,axis=-1,keepdims=True))(input_feature)
    attention=Concatenate(axis=-1)([avg_pool,max_pool])
    attention=Conv2D(filters=1,kernel_size=7,kernel_initializer="he_normal",activation="sigmoid",padding="same")(attention)
    attention=Multiply()([attention,input_feature])
    return attention
def conv_attention_model(input_feature):
    attention=channel_attention(input_feature)
    output=spatial_attention(attention)
    return output
def residual_block(input_feature):
    x=Conv2D(filters=4,kernel_size=1,kernel_initializer="he_normal",padding="same")(input_feature)
    x=Conv2D(filters=4,kernel_size=3,kernel_initializer="he_normal",padding="same")(x)
    x=Conv2D(filters=input_feature.shape[-1],kernel_size=1,kernel_initializer="he_normal")(x)
    return x
def Se_ResNext(input):
    total_res=[]
    for _ in range(32):
        x=residual_block(input)
        total_res.append(x)
    output=Add()([x for x in total_res])
    output=conv_attention_model(output)
    output=Add()([output,input])
    return output
def conv_block(input, num_filters,stride):
    x = Conv2D(num_filters, kernel_size=3, strides=stride,padding="same",kernel_initializer="he_normal",activation="relu")(input)
    x = BatchNormalization()(x)
    return x
def decoder(input):
    x=Conv2DTranspose(filters=128,kernel_size=3,padding="same",strides=2,kernel_initializer="he_normal")(input)
    x=Conv2D(32,kernel_size=3,activation="relu",kernel_initializer="he_normal",padding="same")(x)
    x=BatchNormalization()(x)
    x=Se_ResNext(x)
    skip_1=x
    return x,skip_1
def u_net():
    input = Input(shape=(128, 128, 3))
    output=conv_block(input,64,2)
    output=MaxPooling2D()(output)
    output=conv_block(output,256,1)
    skip_connection_1=output
    output=conv_block(output,512,2)
    skip_connection_2=output
    output=conv_block(output,1024,2)
    skip_connection_3=output
    output=conv_block(output,2048,2)
    skip_connection_4=output
    output=conv_block(output,1024,1)
    output=Concatenate(axis=3)([output,skip_connection_4])
    output,skip_1=decoder(output)
    output=Concatenate(axis=-1)([output,skip_connection_3])
    output,skip_2=decoder(output)
    output=Concatenate(axis=-1)([output,skip_connection_2])
    output,skip_3=decoder(output)
    output=Concatenate(axis=-1)([output,skip_connection_1])
    output,skip_4=decoder(output)
    output,_=decoder(output)
    skip_4=Conv2DTranspose(filters=32,kernel_size=3,strides=2,kernel_initializer="he_normal",padding="same")(skip_4)
    skip_3=Conv2DTranspose(filters=32,kernel_size=3,strides=4,kernel_initializer="he_normal",padding="same")(skip_3)
    skip_2=Conv2DTranspose(filters=32,kernel_size=3,strides=8,kernel_initializer="he_normal",padding="same")(skip_2)
    skip_1=Conv2DTranspose(filters=32,kernel_size=3,strides=16,kernel_initializer="he_normal",padding="same")(skip_1)
    output=Concatenate(axis=-1)([output,skip_4,skip_3,skip_2,skip_1])
    output=conv_block(output,128,1)
    output=Conv2D(filters=1,kernel_size=1,padding="same",kernel_initializer="he_normal",activation="sigmoid")(output)
    model=Model(inputs=input,outputs=output)
    return model
model=u_net()
plot_model(model,show_layer_names=True)
