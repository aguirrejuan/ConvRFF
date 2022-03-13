
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers

from convRFF import ConvRFF
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


def get_model(input_shape=(128,128,3),name='FCNConvRFF',phi_units=2,**kwargs):

    # Encoder 
    input = layers.Input(shape=(128,128,3))

    x =  layers.BatchNormalization()(input)
    
    x =  DefaultConv2D(32)(x)
    x =  DefaultConv2D(32)(x)
    x =  layers.BatchNormalization()(x)
    x = DefaultPooling()(x) # 128x128 -> 64x64

    x =  DefaultConv2D(32)(x)
    x =  DefaultConv2D(32)(x)
    x =  layers.BatchNormalization()(x)
    x = DefaultPooling()(x) # 64x64 -> 32x32

    x =  DefaultConv2D(64)(x)
    x =  DefaultConv2D(64)(x)
    x =  layers.BatchNormalization()(x)
    x = level_1 = DefaultPooling()(x) # 32x32 -> 16x16

    x =  DefaultConv2D(128)(x)
    x =  DefaultConv2D(128)(x)
    x =  layers.BatchNormalization()(x)
    x = level_2 = DefaultPooling()(x) # 16x16 -> 8x8

    x =  DefaultConv2D(256)(x)
    x =  DefaultConv2D(256)(x)
    x =  layers.BatchNormalization()(x)
    x =  DefaultPooling()(x) # 8x8 -> 4x4

    x = DefaultConvRFF(1)(x) 

    x = level_3 = DefaultTranspConv(1,kernel_size=4,use_bias=False)(x)
    x = DefaultConv2D(1,kernel_size=1,activation=None)(level_2)


    x =  layers.Add()([x,level_3])

    
    x = level_4 = DefaultTranspConv(1,kernel_size=4,use_bias=False)(x)
    x = DefaultConv2D(1,kernel_size=1,activation=None)(level_1)

    x =  layers.Add()([x,level_4])

    x = DefaultTranspConv(1,kernel_size=16,strides=8,activation='sigmoid',use_bias=True)(x)


    model = Model(input,x)

    return model 


if __name__ == '__main__':
    model = get_model()
    model.summary()