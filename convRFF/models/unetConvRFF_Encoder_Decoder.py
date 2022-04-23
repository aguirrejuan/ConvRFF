from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers
import tensorflow as tf 

from convRFF.models.convRFF import ConvRFF
from convRFF.models.RFF import RFF
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

upsample = partial(layers.UpSampling2D, (2,2))

def kernel_initializer(seed):
    return tf.keras.initializers.GlorotUniform(seed=seed)

def get_model(input_shape=(128,128,3),name='UnetConvRFF_Decoder_Encoder',
                phi_units=64,padding='SAME', 
                normalization=False,
                kernel_regularizer=regularizers.l2(1e-4), 
                type_layer='cRFF',
                trainable_scale=True,
                kernel_size=3, 
                trainable_W=True,**kwargs):

    # Encoder 
    input = layers.Input(shape=input_shape)

    x =  layers.BatchNormalization()(input)
  
    x =  DefaultConv2D(8,kernel_initializer=kernel_initializer(34),name='Conv10')(x)
    x =  layers.BatchNormalization(name='Batch10')(x)
    x = DefaultConv2D(8,kernel_initializer=kernel_initializer(4),name='Conv11')(x)
    level_1 = DefaultConvRFF(8,trainable_scale=trainable_scale,
                            normalization=normalization,
                            kernel_regularizer=kernel_regularizer,
                            kernel_size=kernel_size, 
                            padding=padding,
                            trainable_W=trainable_W,name='ConvRFF10')(x)
    x =  layers.BatchNormalization(name='Batch11')(x)
    x = DefaultPooling()(x) # 128x128 -> 64x64

    x =  DefaultConv2D(16,kernel_initializer=kernel_initializer(56),name='Conv20')(x)
    x =  layers.BatchNormalization(name='Batch20')(x)
    x = DefaultConv2D(16,kernel_initializer=kernel_initializer(32),name='Conv21')(x)
    level_2 = DefaultConvRFF(16,trainable_scale=trainable_scale,
                            normalization=normalization,
                            kernel_regularizer=kernel_regularizer,
                            kernel_size=kernel_size, 
                            padding=padding,
                            trainable_W=trainable_W,name='ConvRFF20')(x)
    x =  layers.BatchNormalization(name='Batch22')(x)
    x = DefaultPooling()(x) # 64x64 -> 32x32


    x =  DefaultConv2D(32,kernel_initializer=kernel_initializer(87),name='Conv30')(x)
    x =  layers.BatchNormalization(name='Batch30')(x)
    x = DefaultConv2D(32,kernel_initializer=kernel_initializer(30),name='Conv31')(x)
    level_3 = DefaultConvRFF(32,trainable_scale=trainable_scale,
                            normalization=normalization,
                            kernel_regularizer=kernel_regularizer,
                            kernel_size=kernel_size, 
                            padding=padding,
                            trainable_W=trainable_W,name='ConvRFF30')(x)
    x =  layers.BatchNormalization(name='Batch31')(x)
    x = DefaultPooling()(x) # 32x32 -> 16x16

    x = DefaultConv2D(64,kernel_initializer=kernel_initializer(79),name='Conv40')(x)
    x =  layers.BatchNormalization(name='Batch40')(x)
    x =  DefaultConv2D(64,kernel_initializer=kernel_initializer(81),name='Conv41')(x)
    level_4 = DefaultConvRFF(8,trainable_scale=trainable_scale,
                            normalization=normalization,
                            kernel_regularizer=kernel_regularizer,
                            kernel_size=kernel_size, 
                            padding=padding,
                            trainable_W=trainable_W,name='ConvRFF40')(x)
    x =  layers.BatchNormalization(name='Batch41')(x)
    x =  DefaultPooling()(x) # 16x16 -> 8x8


    scale = 16
    
    if type_layer=='cRFF':
        x = DefaultConvRFF(phi_units,
                            trainable_scale=trainable_scale,
                            normalization=normalization,
                            kernel_regularizer=kernel_regularizer,
                            kernel_size=kernel_size, 
                            padding=padding,
                            trainable_W=trainable_W,
                            name='RFF')(x) 
    elif type_layer == 'RFF':
        x = RFF(x,input_shape[0],input_shape[1],phi_units,scale,trainable=trainable_scale,name='RFF')
    
    else:
        x = layers.Lambda(lambda x: x,name='RFF')(x)
        

    x = layers.Reshape((int(input_shape[0]/scale),int(input_shape[1]/scale),-1),name='Reshape')(x)

    #Decoder
    x = DefaultConv2D(128,kernel_initializer=kernel_initializer(89),name='Conv50')(x)
    x =  layers.BatchNormalization(name='Batch50')(x)
    x = DefaultConv2D(128,kernel_initializer=kernel_initializer(42),name='Conv51')(x)
    x =  layers.BatchNormalization(name='Batch51')(x)

    
    x = upsample()(x) # 8x8 -> 16x16
    x = layers.Concatenate()([level_4,x])
    x = DefaultConv2D(64,kernel_initializer=kernel_initializer(91),name='Conv60')(x)
    x =  layers.BatchNormalization(name='Batch60')(x)
    x = DefaultConv2D(64,kernel_initializer=kernel_initializer(47),name='Conv61')(x)
    x =  layers.BatchNormalization(name='Batch61')(x)
    
    x = upsample()(x) # 16x16 -> 32x32
    x = layers.Concatenate()([level_3,x])
    x = DefaultConv2D(32,kernel_initializer=kernel_initializer(21),name='Conv70')(x)
    x =  layers.BatchNormalization(name='Batch70')(x)
    x = DefaultConv2D(32,kernel_initializer=kernel_initializer(96),name='Conv71')(x)
    x =  layers.BatchNormalization(name='Batch71')(x)

    x = upsample()(x) # 32x32 -> 64x64
    x = layers.Concatenate()([level_2,x])
    x = DefaultConv2D(16,kernel_initializer=kernel_initializer(96),name='Conv80')(x)
    x =  layers.BatchNormalization(name='Batch80')(x)
    x = DefaultConv2D(16,kernel_initializer=kernel_initializer(98),name='Conv81')(x)
    x =  layers.BatchNormalization(name='Batch81')(x)

    x = upsample()(x) # 64x64 -> 128x128
    x = layers.Concatenate()([level_1,x])
    x = DefaultConv2D(8,kernel_initializer=kernel_initializer(35),name='Conv90')(x)
    x =  layers.BatchNormalization(name='Batch90')(x)
    x = DefaultConv2D(8,kernel_initializer=kernel_initializer(7),name='Conv91')(x)
    x =  layers.BatchNormalization(name='Batch91')(x)

    x = DefaultConv2D(1,kernel_size=(1,1),activation='sigmoid',kernel_initializer=kernel_initializer(42),name='Conv100')(x)

    model = Model(input,x,name=name)

    return model 



if __name__ == '__main__':
    model = get_model()
    model.summary()
