"""
https://github.com/cralji/RFF-Nerve-UTP/blob/main/FCN_Nerve-UTP.ipynb
"""

from functools import partial
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from convRFF.models import ConvRFF_block



DefaultConv2D = partial(layers.Conv2D,
                        kernel_size=3, activation='relu', padding="same")

DefaultPooling = partial(layers.MaxPool2D,
                        pool_size=2)

DefaultTranspConv = partial(layers.Conv2DTranspose,
                            kernel_size=3, strides=2,
                            padding='same',
                            use_bias=False, activation='relu')


def kernel_initializer(seed):
    return tf.keras.initializers.GlorotUniform(seed=seed)


def get_model(input_shape=(128,128,3), name='FCN_RFF', out_channels=1, 
              out_ActFunction='sigmoid',
                kernel_regularizer=None, **kwargs_convrff):
    # Encoder 
    k_r = kernel_regularizer#regularizers.L1L2(l1=1e-5, l2=1e-4)
    input_ = layers.Input(shape=input_shape)

    x =  layers.BatchNormalization(name='Batch00')(input_)
    
    x =  ConvRFF_block(x, deepth=16,name='10',kernel_regularizer=k_r,**kwargs_convrff)
    x =  ConvRFF_block(x, deepth=16,name='11',kernel_regularizer=k_r,**kwargs_convrff)
    x =  layers.BatchNormalization(name='Batch10')(x)
    x = DefaultPooling(name='Pool10')(x) # 128x128 -> 64x64

    x =  ConvRFF_block(x, deepth=16,name='20',kernel_regularizer=k_r,**kwargs_convrff)
    x =  ConvRFF_block(x, deepth=16,name='21',kernel_regularizer=k_r,**kwargs_convrff)
    x =  layers.BatchNormalization(name='Batch20')(x)
    x = DefaultPooling(name='Pool20')(x) # 64x64 -> 32x32

    x =  ConvRFF_block(x, deepth=32,name='30',kernel_regularizer=k_r,**kwargs_convrff)
    x =  ConvRFF_block(x, deepth=32,name='31',kernel_regularizer=k_r,**kwargs_convrff)
    x =  layers.BatchNormalization(name='Batch30')(x)
    x = level_1 = DefaultPooling(name='Pool30')(x) # 32x32 -> 16x16

    x =  ConvRFF_block(x, deepth=64,name='40',kernel_regularizer=k_r,**kwargs_convrff)
    x =  ConvRFF_block(x, deepth=64,name='41',kernel_regularizer=k_r,**kwargs_convrff)
    x =  layers.BatchNormalization(name='Batch40')(x)
    x = level_2 = DefaultPooling(name='Pool40')(x) # 16x16 -> 8x8

    x =  ConvRFF_block(x, deepth=128,name='50',kernel_regularizer=k_r,**kwargs_convrff)
    x =  ConvRFF_block(x, deepth=128,name='51',kernel_regularizer=k_r,**kwargs_convrff)
    x =  layers.BatchNormalization(name='Batch50')(x)
    x =  DefaultPooling(name='Pool50')(x) # 8x8 -> 4x4

    #Decoder
    x = level_3 = DefaultTranspConv(out_channels,kernel_size=4,
                                    use_bias=False, 
                                    kernel_initializer=kernel_initializer(98),
                                    name='Trans60')(x)
    x = DefaultConv2D(out_channels,kernel_size=1,
                    activation=None,kernel_initializer=kernel_initializer(75),
                    name='Conv60')(level_2)


    x =  layers.Add(name='Add10')([x,level_3])

    
    x = level_4 = DefaultTranspConv(out_channels,kernel_size=4,use_bias=False,
                                    kernel_initializer=kernel_initializer(87),
                                    name='Trans70')(x)
    x = DefaultConv2D(out_channels,kernel_size=1,activation=None,
                        kernel_initializer=kernel_initializer(54),
                        name='Conv70')(level_1)

    x =  layers.Add(name='Add20')([x,level_4])

    x = DefaultTranspConv(out_channels,kernel_size=16,strides=8,
                            activation=out_ActFunction,use_bias=True,
                            kernel_initializer=kernel_initializer(32),
                            name='Trans80')(x)


    model = Model(input_,x,name=name)

    return model 



if __name__ == "__main__":
    model = fcn_baseline()
    model.summary()