import tensorflow as tf 
from tensorflow.keras import Model, layers, regularizers
from functools import partial

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