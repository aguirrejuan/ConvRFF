import numpy as np
import tensorflow as tf 

def _get_random_features_initializer(initializer, shape):

    def _get_cauchy_samples(loc, scale, shape):
        probs = np.random.uniform(low=0., high=1., size=shape)
        return loc + scale * np.tan(np.pi * (probs - 0.5))

    if isinstance(initializer,str):
        if initializer == "gaussian":
            return tf.keras.initializers.RandomNormal(stddev=1.0)
        elif initializer == "laplacian":
            return tf.keras.initializers.Constant(
                _get_cauchy_samples(loc=0.0, scale=1.0, shape=shape))
        else: 
            raise ValueError(f'Unsupported kernel initializer {initializer}')


class ConvRFF(tf.keras.layers.Layer):

    def __init__(self,output_dim, kernel_size=3,
                 scale=None,
                 trainable_scale=False, trainable_W=False,
                 kernel='gaussian',
                 padding='VALID',
                 stride=1,
                 kernel_regularizer=None,
                 **kwargs):
        
        super(ConvRFF,self).__init__(**kwargs)

        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.scale = scale 
        self.trainable_scale = trainable_scale 
        self.trainable_W = trainable_W
        self.padding = padding
        self.stride = stride
        self.initializer = kernel
        self.kernel_regularizer = kernel_regularizer

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'kernel_size': self.kernel_size,
            'scale': self.scale,
            'trainable_scale': self.trainable_scale,
            'trainable_W':self.trainable_W,
            'padding':self.padding,
            'kernel':self.initializer,
        })
        return config

    def build(self,input_shape):
        
        input_dim = input_shape[-1]

        kernel_initializer = _get_random_features_initializer(self.initializer,
                                                              shape=(self.kernel_size,
                                                                     self.kernel_size,
                                                                     input_dim,
                                                                     self.output_dim))

        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size,self.kernel_size,
                   input_dim,self.output_dim),
            dtype=tf.float32,
            initializer = kernel_initializer,
            trainable =self.trainable_W,
            regularizer = self.kernel_regularizer,
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.output_dim,),
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                mean=0.0,stddev=2*np.pi),
            trainable=self.trainable_W
        )

        if not self.scale:
            if  self.initializer == 'gaussian':
                self.scale = np.sqrt(self.kernel_size**2/2.0)
            elif self.initializer == 'laplacian':
                self.scale = 1.0
            else: 
                raise ValueError(f'Unsupported kernel initializer {self.initializer}')


        self.kernel_scale = self.add_weight(
            name='kernel_scale',
            shape=(1,),
            dtype=tf.float32,
            initializer=tf.compat.v1.constant_initializer(self.scale),
            trainable=self.trainable_scale,
            constraint='NonNeg'
        )
    

    def call(self,inputs):

        scale = tf.math.divide(1.0,  self.kernel_scale)

        kernel = tf.math.multiply(scale,self.kernel)

        outputs = tf.nn.conv2d(inputs,kernel,
                               strides=[1,self.stride,self.stride,1],
                               padding=self.padding)
        outputs = tf.nn.bias_add(outputs,self.bias)

        outputs = tf.cos(outputs)*tf.math.sqrt(2/self.output_dim)
        return outputs