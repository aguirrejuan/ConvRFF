from tensorflow.keras import layers, initializers
from functools import partial
from convRFF.layers import ConvRFF_block

def upsample_conv(filters, kernel_size, strides, padding, kernel_initializer, name, kernel_regularizer):
    return layers.Conv2DTranspose(filters, kernel_size,
                                 strides=strides,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer,
                                 padding=padding,
                                 name=name)



DefaultConv2D = partial(layers.Conv2D,
                        kernel_size=3, activation='relu', padding="same")


def res_block(x,units,kernel_initializer,name,kernel_regularizer=None):
    x_c = x
    x = layers.Conv2D(units,(1,1),(1,1),
                      kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,
                      padding='same',name=f'{name}_Conv00')(x)
    x = layers.BatchNormalization(name=f'{name}_Batch00')(x)
    x = layers.Activation('relu',name=f'{name}_Act00')(x)
    x = layers.Conv2D(units,(3,3),(1,1),
                      kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,
                      padding='same',
                      name=f'{name}_Conv01')(x)
    x = layers.BatchNormalization(name=f'{name}_Batch01')(x)
    x_c = layers.Conv2D(units,(1,1),(1,1),
                        kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,
                        padding='same',
                        name=f'{name}_Conv02')(x_c)
    x_c = layers.BatchNormalization(name=f'{name}_Batch02')(x_c)
    x = layers.Add(name=f'{name}_Add00')([x,x_c])
    x = layers.Activation('relu',name=f'{name}_Act01')(x)
    return x


def res_block_convRFF(x, deepth=16, name='10', 
                        kernel_regularizer=None, **kwargs_convrff):
    x_c = x
    x = ConvRFF_block(x, kernel_size=1, deepth=deepth,
                     name=f'{name}_ConvRFF00',
                     kernel_regularizer=kernel_regularizer,
                     **kwargs_convrff)
    x = layers.BatchNormalization(name=f'{name}_Batch00')(x)
    x = layers.Activation('relu',name=f'{name}_Act00')(x)
    x = ConvRFF_block(x, kernel_size=3, deepth=deepth,
                     name=f'{name}_ConvRFF01',
                     kernel_regularizer=kernel_regularizer,
                     **kwargs_convrff)
    x = layers.BatchNormalization(name=f'{name}_Batch01')(x)
    x_c = ConvRFF_block(x_c, kernel_size=1, deepth=deepth,
                     name=f'{name}_ConvRFF02',
                     kernel_regularizer=kernel_regularizer,
                     **kwargs_convrff)
    x_c = layers.BatchNormalization(name=f'{name}_Batch02')(x_c)
    x = layers.Add(name=f'{name}_Add00')([x,x_c])
    x = layers.Activation('relu',name=f'{name}_Act01')(x)
    return x


def kernel_initializer(seed):
    return initializers.GlorotUniform(seed=seed)