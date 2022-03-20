
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers

from .convRFF import ConvRFF, RFF
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



def get_model(input_shape=(128,128,3),name='FCNConvRFF',phi_units=2,out_channels=1,cRFF=True,kernel_size=3,trainable_scale=True, trainable_W=True,**kwargs):

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

    scale = 32
    x = DefaultConvRFF(phi_units,trainable_scale=trainable_scale,kernel_size=kernel_size, trainable_W=trainable_W)(x) if cRFF else  RFF(x,input_shape[0],input_shape[1],phi_units,scale,trainable=trainable_scale)
    x = layers.Reshape((int(input_shape[0]/scale),int(input_shape[1]/scale),-1))(x)

    x = level_3 = DefaultTranspConv(out_channels,kernel_size=4,use_bias=False)(x)
    x = DefaultConv2D(out_channels,kernel_size=1,activation=None)(level_2)


    x =  layers.Add()([x,level_3])

    
    x = level_4 = DefaultTranspConv(out_channels,kernel_size=4,use_bias=False)(x)
    x = DefaultConv2D(out_channels,kernel_size=1,activation=None)(level_1)

    x =  layers.Add()([x,level_4])

    x = DefaultTranspConv(1,kernel_size=16,strides=8,activation='sigmoid',use_bias=True)(x)


    model = Model(input,x,name=name)

    return model 


if __name__ == '__main__':
    model = get_model()
    model.summary()
