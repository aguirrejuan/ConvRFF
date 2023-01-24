"""
https://github.com/cralji/RFF-Nerve-UTP/blob/main/Res-UNET_Nerve-UTP.ipynb
"""

from tensorflow.keras import Model, layers, initializers
from convRFF.models.res_unet import upsample_conv, DefaultConv2D, kernel_initializer, res_block


def get_model(input_shape=(128,128,3), name='RES_UNET', out_channels=1, out_ActFunction='sigmoid',
            kernel_regularizer=None ):

    # Encoder 
    kr = kernel_regularizer#regularizers.L1L2(l1=1e-5, l2=1e-4)
    input_ = layers.Input(shape=input_shape, name='input')

    pp_in_layer = input_

    pp_in_layer = layers.BatchNormalization()(pp_in_layer)
    c1 = res_block(pp_in_layer,8,kernel_initializer=kernel_initializer(34),kernel_regularizer=kr,name='Res00')
    c1 = res_block(c1,8,kernel_initializer=kernel_initializer(3),kernel_regularizer=kr,name='Res01')
    level_1 =  DefaultConv2D(8,kernel_initializer=kernel_initializer(3321),kernel_regularizer=kr,name='Conv01')(c1)
    p1 = layers.MaxPooling2D((2, 2),name='Maxp00') (c1)

    c2 = res_block(p1,16,kernel_initializer=kernel_initializer(7),kernel_regularizer=kr,name='Res02')
    c2 = res_block(c2,16,kernel_initializer=kernel_initializer(98),kernel_regularizer=kr,name='Res03')
    level_2 =  DefaultConv2D(16,kernel_initializer=kernel_initializer(23),kernel_regularizer=kr,name='Conv02')(c2)
    p2 = layers.MaxPooling2D((2, 2),name='Maxp01') (c2)

    c3 = res_block(p2,32,kernel_initializer=kernel_initializer(5),kernel_regularizer=kr,name='Res04')
    c3 = res_block(c3,32,kernel_initializer=kernel_initializer(23),kernel_regularizer=kr,name='Res05')
    level_3 =  DefaultConv2D(32,kernel_initializer=kernel_initializer(343),kernel_regularizer=kr,name='Conv03')(c3)
    p3 = layers.MaxPooling2D((2, 2),name='Maxp02') (c3)

    c4 = res_block(p3,64,kernel_initializer=kernel_initializer(32),kernel_regularizer=kr,name='Res06')
    c4 = res_block(c4,64,kernel_initializer=kernel_initializer(43),kernel_regularizer=kr,name='Res07')
    level_4 =  DefaultConv2D(64,kernel_initializer=kernel_initializer(65),kernel_regularizer=kr,name='Conv04')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2),name='Maxp03') (c4)

    # Bottle Neck
    c5 = res_block(p4,128,kernel_initializer=kernel_initializer(43),kernel_regularizer=kr,name='Res08')
    c5 = res_block(c5,128,kernel_initializer=kernel_initializer(65),kernel_regularizer=kr,name='Res09')
    # upsampling
    u6 = upsample_conv(64, (2, 2), strides=(2, 2),
                       padding='same',
                       kernel_initializer=kernel_initializer(4),kernel_regularizer=kr,
                       name='Upsam00') (c5)
    u6 = layers.concatenate([u6, level_4],name='Concat00')
    c6 = res_block(u6,64,kernel_initializer=kernel_initializer(65),kernel_regularizer=kr,name='Res10')
    c6 = res_block(c6,64,kernel_initializer=kernel_initializer(87),kernel_regularizer=kr,name='Res11')

    u7 = upsample_conv(32, (2, 2), strides=(2, 2), 
                       padding='same',
                       kernel_initializer=kernel_initializer(2),kernel_regularizer=kr,
                       name='Upsam01') (c6)
    u7 = layers.concatenate([u7, level_3],name='Concat01')
    c7 = res_block(u7,32,kernel_initializer=kernel_initializer(34),kernel_regularizer=kr,name='Res12')
    c7 = res_block(c7,32,kernel_initializer=kernel_initializer(4),kernel_regularizer=kr,name='Res13')

    u8 = upsample_conv(16, (2, 2), strides=(2, 2),
                       padding='same',
                       kernel_initializer=kernel_initializer(432),kernel_regularizer=kr,
                       name='Upsam02') (c7)
    u8 = layers.concatenate([u8, level_2],name='Concat02')
    c8 = res_block(u8,16,kernel_initializer=kernel_initializer(32),kernel_regularizer=kr,name='Res14')
    c8 = res_block(c8,16,kernel_initializer=kernel_initializer(42),kernel_regularizer=kr,name='Res15')

    u9 = upsample_conv(8, (2, 2), strides=(2, 2), 
                       padding='same',
                       kernel_initializer=kernel_initializer(32),kernel_regularizer=kr,
                       name='Upsam03') (c8)
    u9 = layers.concatenate([u9, level_1], axis=3,name='Concat03')
    c9 = res_block(u9,8,kernel_initializer=kernel_initializer(4),kernel_regularizer=kr,name='Res16')
    c9 = res_block(c9,8,kernel_initializer=kernel_initializer(6),kernel_regularizer=kr,name='Res17')

    d = layers.Conv2D(out_channels, kernel_size=(1, 1),kernel_regularizer=kr, activation=out_ActFunction,name='Output') (c9)
    
    seg_model = Model(inputs=[input_], outputs=[d])
    
    return seg_model

if __name__ == '__main__':
    from tensorflow.keras import regularizers
    kernel_regularizer = regularizers.L1L2(l1=1e-5, l2=1e-4)
    model = get_model(kernel_regularizer=kernel_regularizer)
    model.summary()