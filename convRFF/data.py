import tensorflow as tf 
import tensorflow_addons as tfa
import numpy as np 


def resize(shape=(256,256)):
    def func(img,mask):
        return tf.image.resize(img,shape),tf.image.resize(mask,shape)
    return func


def data_augmentation_func(flip_left_right=True, 
                            flip_up_down=True, range_rotate=(-10,10)):
    def data_aug(img, mask):
        seed = tf.random.uniform(shape=(2,), minval=1, maxval=1000, dtype=tf.int32)

        if flip_left_right:
            img = tf.image.stateless_random_flip_left_right(img, seed)
            mask = tf.image.stateless_random_flip_left_right(mask, seed)

        if flip_up_down:
            img =  tf.image.stateless_random_flip_up_down(img, seed)
            mask = tf.image.stateless_random_flip_up_down(mask, seed)

        if range_rotate:
            min_val = range_rotate[0]/180*np.pi
            max_val = range_rotate[1]/180*np.pi
            rotation = tf.random.uniform(shape=(), minval=min_val, maxval=max_val, dtype=tf.float32)
            img = tfa.image.rotate(img, rotation, fill_mode='reflect')
            mask = tfa.image.rotate(mask, rotation, fill_mode='constant')
        return img, mask

    return data_aug


def preprocess_data(data, data_augmentation=False, 
                    return_label_info=False, shape=256, repeat=1,
                    flip_left_right=True, 
                    flip_up_down=True,
                    range_rotate=(-10,10)):

    if not return_label_info:
        data = data.map(lambda *items: items[:2],
                        num_parallel_calls=tf.data.AUTOTUNE)

    data = data.map(lambda x,y,*l: (*resize((shape,shape))(x,y),*l),
                     num_parallel_calls=tf.data.AUTOTUNE)
    data = data.cache()

    data = data.repeat(repeat)
    if data_augmentation:
        data = data.map(
                        data_augmentation_func(
                                    flip_left_right=flip_left_right, 
                                    flip_up_down=flip_up_down,
                                    range_rotate=range_rotate
                                    ), 
                        num_parallel_calls=tf.data.AUTOTUNE
                        )

    return data



def get_data(dataset_class, seed=42, 
             data_augmentation=True,
             return_label_info=False, 
             shape=256, batch_size=32,
             repeat=1,
             flip_left_right=True, 
             flip_up_down=True,
             range_rotate=(-10,10)):

    dataset = dataset_class(seed=seed)
    train_data, val_data, test_data = dataset()

    train_data = preprocess_data(
                    train_data, data_augmentation, 
                    return_label_info, shape, repeat=repeat, 
                    flip_left_right=flip_left_right, 
                    flip_up_down=flip_up_down,
                    range_rotate=range_rotate
                    ).batch(batch_size=batch_size)

    val_data = preprocess_data(
                                val_data, False, 
                                return_label_info, 
                                shape
                                ).batch(batch_size=batch_size)

    test_data = preprocess_data(
                                test_data, False, 
                                return_label_info,
                                shape
                                ).batch(batch_size=batch_size)
            
    return train_data, val_data, test_data