"""
https://github.com/cralji/RFF-Nerve-UTP/blob/main/Res-UNET_Nerve-UTP.ipynb
"""

from tensorflow.keras import Model, layers, initializers
from convRFF.models.res_unet import upsample_conv, DefaultConv2D, kernel_initializer, res_block_convRFF, res_block


def get_model(input_shape=(128,128,3), 
             name='RES_UNET_RFF', out_channels=1,
              out_ActFunction='sigmoid',
              kernel_regularizer=None, **kwargs_convrff):


    k_r = kernel_regularizer#regularizers.L1L2(l1=1e-5, l2=1e-4)

    input_ = layers.Input(shape=input_shape, name='input')
    pp_in_layer = input_

    pp_in_layer = layers.BatchNormalization()(pp_in_layer)
    c1 = res_block_convRFF(pp_in_layer,4,name='Res00',kernel_regularizer=k_r)
    c1 = res_block_convRFF(c1,4,name='Res01',kernel_regularizer=k_r)
    p1 = layers.MaxPooling2D((2, 2),name='Maxp00') (c1)

    c2 = res_block_convRFF(p1,8,name='Res02',kernel_regularizer=k_r)
    c2 = res_block_convRFF(c2,8,name='Res03',kernel_regularizer=k_r)
    p2 = layers.MaxPooling2D((2, 2),name='Maxp01') (c2)

    c3 = res_block_convRFF(p2,16,name='Res04',kernel_regularizer=k_r)
    c3 = res_block_convRFF(c3,16,name='Res05',kernel_regularizer=k_r)
    p3 = layers.MaxPooling2D((2, 2),name='Maxp02') (c3)

    c4 = res_block_convRFF(p3,32,name='Res06',kernel_regularizer=k_r)
    c4 = res_block_convRFF(c4,32,name='Res07',kernel_regularizer=k_r)
    p4 = layers.MaxPooling2D(pool_size=(2, 2),name='Maxp03') (c4)
    # Bottle Neck
    c5 = res_block(p4,128,kernel_initializer=kernel_initializer(43),kernel_regularizer=k_r,name='Res08')
    c5 = res_block(c5,128,kernel_initializer=kernel_initializer(65),kernel_regularizer=k_r,name='Res09')
    # upsampling
    u6 = upsample_conv(64, (2, 2), strides=(2, 2),
                       padding='same',
                       kernel_initializer=kernel_initializer(4),kernel_regularizer=k_r,
                       name='Upsam00') (c5)
    u6 = layers.concatenate([u6, c4],name='Concat00')
    c6 = res_block(u6,64,kernel_initializer=kernel_initializer(65),kernel_regularizer=k_r,name='Res10')
    c6 = res_block(c6,64,kernel_initializer=kernel_initializer(87),kernel_regularizer=k_r,name='Res11')

    u7 = upsample_conv(32, (2, 2), strides=(2, 2), 
                       padding='same',
                       kernel_initializer=kernel_initializer(2),kernel_regularizer=k_r,
                       name='Upsam01') (c6)
    u7 = layers.concatenate([u7, c3],name='Concat01')
    c7 = res_block(u7,32,kernel_initializer=kernel_initializer(34),kernel_regularizer=k_r,name='Res12')
    c7 = res_block(c7,32,kernel_initializer=kernel_initializer(4),kernel_regularizer=k_r,name='Res13')

    u8 = upsample_conv(16, (2, 2), strides=(2, 2),
                       padding='same',
                       kernel_initializer=kernel_initializer(432),kernel_regularizer=k_r,
                       name='Upsam02') (c7)
    u8 = layers.concatenate([u8, c2],name='Concat02')
    c8 = res_block(u8,16,kernel_initializer=kernel_initializer(32),kernel_regularizer=k_r,name='Res14')
    c8 = res_block(c8,16,kernel_initializer=kernel_initializer(42),kernel_regularizer=k_r,name='Res15')

    u9 = upsample_conv(8, (2, 2), strides=(2, 2), 
                       padding='same',
                       kernel_initializer=kernel_initializer(32),kernel_regularizer=k_r,
                       name='Upsam03') (c8)
    u9 = layers.concatenate([u9, c1], axis=3,name='Concat03')
    c9 = res_block(u9,8,kernel_initializer=kernel_initializer(4),kernel_regularizer=k_r,name='Res16')
    c9 = res_block(c9,8,kernel_initializer=kernel_initializer(6),kernel_regularizer=k_r,name='Res17')

    d = layers.Conv2D(out_channels, kernel_size=(1, 1),kernel_regularizer=k_r, activation=out_ActFunction,name='Output') (c9)
    
    seg_model = Model(inputs=[input_], outputs=[d])
    
    return seg_model

if __name__ == '__main__':
    from tensorflow.keras import regularizers
    kernel_regularizer =regularizers.L1L2(l1=1e-5, l2=1e-4)
    model = get_model(kernel_regularizer=kernel_regularizer)
    model.summary()