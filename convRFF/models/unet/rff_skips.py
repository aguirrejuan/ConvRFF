"""
https://github.com/cralji/RFF-Nerve-UTP/blob/main/UNET-Nerve-UTP.ipynb
"""

from functools import partial
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from convRFF.layers import ConvRFF_block



DefaultConv2D = partial(layers.Conv2D,
                        kernel_size=3, activation='relu', padding="same")

DefaultPooling = partial(layers.MaxPool2D,
                        pool_size=2)

upsample = partial(layers.UpSampling2D, (2,2))

def kernel_initializer(seed):
    return tf.keras.initializers.GlorotUniform(seed=seed)


def get_model(input_shape=(128,128,3), name='rff_skips', 
              out_channels=1, kernel_regularizer=None, **kwargs_convrff):

    # Encoder 
    k_r = kernel_regularizer#regularizers.L1L2(l1=1e-5, l2=1e-4)
    input_ = layers.Input(shape=input_shape)

    x =  layers.BatchNormalization(name='Batch00')(input_)
  
    x =  ConvRFF_block(x,deepth=4,name='00',kernel_regularizer=k_r,**kwargs_convrff)
    x =  layers.BatchNormalization(name='Batch10')(x)
    x = level_1 = ConvRFF_block(x,deepth=4,name='11',kernel_regularizer=k_r,**kwargs_convrff)
    x =  layers.BatchNormalization(name='Batch11')(x)
    x = DefaultPooling(name='Pool10')(x) # 128x128 -> 64x64

    level_1 = ConvRFF_block(level_1,deepth=8,name='01',kernel_regularizer=k_r,**kwargs_convrff)

    x = ConvRFF_block(x,deepth=8,name='20',kernel_regularizer=k_r,**kwargs_convrff)
    x =  layers.BatchNormalization(name='Batch20')(x)
    x = level_2 = ConvRFF_block(x,deepth=8,name='21',kernel_regularizer=k_r,**kwargs_convrff)
    x =  layers.BatchNormalization(name='Batch22')(x)
    x = DefaultPooling(name='Pool20')(x) # 64x64 -> 32x32

    level_2 = ConvRFF_block(level_2,deepth=16, name='02',kernel_regularizer=k_r,**kwargs_convrff)

    x =  ConvRFF_block(x,deepth=16,name='30',kernel_regularizer=k_r,**kwargs_convrff)
    x =  layers.BatchNormalization(name='Batch30')(x)
    x = level_3 = ConvRFF_block(x,deepth=16,name='31',kernel_regularizer=k_r,**kwargs_convrff)
    x =  layers.BatchNormalization(name='Batch31')(x)
    x = DefaultPooling(name='Pool30')(x) # 32x32 -> 16x16

    level_3 = ConvRFF_block(level_3,deepth=32, name='03',kernel_regularizer=k_r,**kwargs_convrff)

    x = ConvRFF_block(x,deepth=32,name='40',kernel_regularizer=k_r,**kwargs_convrff)
    x =  layers.BatchNormalization(name='Batch40')(x)
    x = level_4 =  ConvRFF_block(x,deepth=32,name='41',kernel_regularizer=k_r,**kwargs_convrff) 
    x =  layers.BatchNormalization(name='Batch41')(x)
    x =  DefaultPooling(name='Pool40')(x) # 16x16 -> 8x8

    level_4 = ConvRFF_block(level_4,deepth=64,name='04',kernel_regularizer=k_r,**kwargs_convrff)

    #Decoder
    x = DefaultConv2D(128,kernel_initializer=kernel_initializer(89),name='Conv50')(x)
    x =  layers.BatchNormalization(name='Batch50')(x)
    x = DefaultConv2D(128,kernel_initializer=kernel_initializer(42),name='Conv51')(x)
    x =  layers.BatchNormalization(name='Batch51')(x)

    
    x = upsample(name='Up60')(x) # 8x8 -> 16x16
    x = layers.Concatenate()([level_4,x])
    x = DefaultConv2D(64,kernel_initializer=kernel_initializer(91),name='Conv60')(x)
    x =  layers.BatchNormalization(name='Batch60')(x)
    x = DefaultConv2D(64,kernel_initializer=kernel_initializer(47),name='Conv61')(x)
    x =  layers.BatchNormalization(name='Batch61')(x)
    
    x = upsample(name='Up70')(x) # 16x16 -> 32x32
    x = layers.Concatenate()([level_3,x])
    x = DefaultConv2D(32,kernel_initializer=kernel_initializer(21),name='Conv70')(x)
    x =  layers.BatchNormalization(name='Batch70')(x)
    x = DefaultConv2D(32,kernel_initializer=kernel_initializer(96),name='Conv71')(x)
    x =  layers.BatchNormalization(name='Batch71')(x)

    x = upsample(name='Up80')(x) # 32x32 -> 64x64
    x = layers.Concatenate()([level_2,x])
    x = DefaultConv2D(16,kernel_initializer=kernel_initializer(96),name='Conv80')(x)
    x =  layers.BatchNormalization(name='Batch80')(x)
    x = DefaultConv2D(16,kernel_initializer=kernel_initializer(98),name='Conv81')(x)
    x =  layers.BatchNormalization(name='Batch81')(x)

    x = upsample(name='Up90')(x) # 64x64 -> 128x128
    x = layers.Concatenate()([level_1,x])
    x = DefaultConv2D(8,kernel_initializer=kernel_initializer(35),name='Conv90')(x)
    x =  layers.BatchNormalization(name='Batch90')(x)
    x = DefaultConv2D(8,kernel_initializer=kernel_initializer(7),name='Conv91')(x)
    x =  layers.BatchNormalization(name='Batch91')(x)

    x = DefaultConv2D(out_channels,kernel_size=(1,1),activation='sigmoid',
                        kernel_initializer=kernel_initializer(42),
                        name='Conv100')(x)

    model = Model(input_,x,name=name)

    return model 



if __name__ == '__main__':
    model = get_model()
    model.summary()
