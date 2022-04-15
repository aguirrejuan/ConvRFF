from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers
import tensorflow as tf 

from convRFF.models.convRFF import ConvRFF, RFF
from functools import partial

DefaultConv2D = partial(layers.Conv2D,
                        kernel_size=3, activation='relu', padding="same")

DefaultPooling = partial(layers.MaxPool2D,
                        pool_size=2)

DefaultConvRFF = partial(ConvRFF,
                        kernel_size=3, padding="SAME",
                         kernel_regularizer = regularizers.l2(1e-4),
                        trainable_scale=True, trainable_W=True,
                         )

DefaultTranspConv = partial(layers.Conv2DTranspose,
                            kernel_size=3, strides=2,
                            padding='same',
                            use_bias=False, activation='relu')



def kernel_initializer(seed):
    return tf.keras.initializers.GlorotUniform(seed=seed)

def get_model(input_shape=(128,128,3),name='FCNConvRFF',kernel_regularizer=regularizers.l2(1e-4),normalization=False,phi_units=2,out_channels=1,type_layer='cRFF',padding='SAME',kernel_size=3,trainable_scale=True, trainable_W=True,**kwargs):

    # Encoder 
    input = layers.Input(shape=(128,128,3))

    x =  layers.BatchNormalization()(input)
    
    x =  DefaultConv2D(32,kernel_initializer=kernel_initializer(34),name='Conv10')(x)
    x =  DefaultConv2D(32,kernel_initializer=kernel_initializer(4),name='Conv11')(x)
    x =  layers.BatchNormalization(name='Batch10')(x)
    x = DefaultPooling()(x) # 128x128 -> 64x64

    x =  DefaultConv2D(32,kernel_initializer=kernel_initializer(56),name='Conv20')(x)
    x =  DefaultConv2D(32,kernel_initializer=kernel_initializer(28),name='Conv21')(x)
    x =  layers.BatchNormalization(name='Batch20')(x)
    x = DefaultPooling()(x) # 64x64 -> 32x32

    x =  DefaultConv2D(64,kernel_initializer=kernel_initializer(332),name='Conv30')(x)
    x =  DefaultConv2D(64,kernel_initializer=kernel_initializer(2),name='Conv31')(x)
    x =  layers.BatchNormalization(name='Batch30')(x)
    x = level_1 = DefaultPooling()(x) # 32x32 -> 16x16

    x =  DefaultConv2D(128,kernel_initializer=kernel_initializer(67),name='Conv40')(x)
    x =  DefaultConv2D(128,kernel_initializer=kernel_initializer(89),name='Conv41')(x)
    x =  layers.BatchNormalization(name='Batch40')(x)
    x = level_2 = DefaultPooling()(x) # 16x16 -> 8x8

    x =  DefaultConv2D(256,kernel_initializer=kernel_initializer(7),name='Conv50')(x)
    x =  DefaultConv2D(256,kernel_initializer=kernel_initializer(23),name='Conv51')(x)
    x =  layers.BatchNormalization(name='Batch50')(x)
    x =  DefaultPooling()(x) # 8x8 -> 4x4

    scale = 32
    if type_layer == 'cRFF':
        x = DefaultConvRFF(phi_units,
                            trainable_scale=trainable_scale,
                            normalization=normalization,
                            kernel_regularizer=kernel_regularizer,
                            kernel_size=kernel_size,
                            padding=padding,
                            trainable_W=trainable_W,
                            name='ConvRFF')(x) 
    elif type_layer=='RFF':
        x = RFF(x,input_shape[0],input_shape[1],phi_units,scale,trainable=trainable_scale)
    else: 
        x = x 

    x = layers.Reshape((int(input_shape[0]/scale),int(input_shape[1]/scale),-1),name='Reshape')(x)

    x = level_3 = DefaultTranspConv(out_channels,kernel_size=4,use_bias=False, kernel_initializer=kernel_initializer(98),name='Trans60')(x)
    x = DefaultConv2D(out_channels,kernel_size=1,activation=None,kernel_initializer=kernel_initializer(75),name='Conv60')(level_2)


    x =  layers.Add()([x,level_3])

    
    x = level_4 = DefaultTranspConv(out_channels,kernel_size=4,use_bias=False,kernel_initializer=kernel_initializer(87),name='Trans70')(x)
    x = DefaultConv2D(out_channels,kernel_size=1,activation=None,kernel_initializer=kernel_initializer(54),name='Conv70')(level_1)

    x =  layers.Add()([x,level_4])

    x = DefaultTranspConv(1,kernel_size=16,strides=8,activation='sigmoid',use_bias=True,kernel_initializer=kernel_initializer(32),name='Trans80')(x)


    model = Model(input,x,name=name)

    return model 


if __name__ == '__main__':
    model = get_model()
    model.summary()
