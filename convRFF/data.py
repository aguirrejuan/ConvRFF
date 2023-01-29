import tensorflow as tf 
import numpy as np 


def resize(shape=(256,256)):
    def func(img,mask):
        return tf.image.resize(img,shape),tf.image.resize(mask,shape)
    return func


def random_translation(img,mask,translation_h_w):
    translation1 = tf.keras.layers.RandomTranslation(height_factor=translation_h_w[0],
                                                     width_factor=translation_h_w[1], 
                                                     seed=42, fill_mode='nearest')

    translation2 = tf.keras.layers.RandomTranslation(height_factor=translation_h_w[0],
                                                     width_factor=translation_h_w[1], 
                                                     seed=42, fill_mode='constant')

    img = translation1(img)
    mask = translation2(mask)

    return img, mask 


def random_zoom(img, mask, zoom_h_w):
    zoom1 = tf.keras.layers.RandomZoom(height_factor=zoom_h_w[0],
                                                     width_factor=zoom_h_w[1], 
                                                     seed=42, fill_mode='nearest')

    zoom2 = tf.keras.layers.RandomZoom(height_factor=zoom_h_w[0],
                                                     width_factor=zoom_h_w[1], 
                                                     seed=42, fill_mode='constant')
    img = zoom1(img)
    mask = zoom2(mask)
    return img, mask



def random_rotation(img,mask,range_rotate):
    min_val = range_rotate[0]/180*np.pi
    max_val = range_rotate[1]/180*np.pi
    range_rotate = [min_val,max_val]
    img = tf.keras.layers.RandomRotation(range_rotate,fill_mode='reflect',
                                         interpolation='bilinear',
                                         seed=42,
                                         fill_value=0.0)(img)
    mask = tf.keras.layers.RandomRotation(range_rotate,fill_mode='constant',
                                         interpolation='bilinear',
                                         seed=42,
                                         fill_value=0.0)(mask)

    return img, mask
    



def data_augmentation_func(flip_left_right=True, 
                            flip_up_down=True, range_rotate=(-10,10), 
                            translation_h_w=None, zoom_h_w=None):
    def data_aug(img, mask):
        seed = tf.random.uniform(shape=(2,), minval=1, maxval=1000, dtype=tf.int32)

        if flip_left_right:
            seed = tf.random.uniform(shape=(2,), minval=1, maxval=1000, dtype=tf.int32)
            img = tf.image.stateless_random_flip_left_right(img, seed)
            mask = tf.image.stateless_random_flip_left_right(mask, seed)

        if flip_up_down:
            seed = tf.random.uniform(shape=(2,), minval=1, maxval=1000, dtype=tf.int32)
            img =  tf.image.stateless_random_flip_up_down(img, seed)
            mask = tf.image.stateless_random_flip_up_down(mask, seed)

        if range_rotate:
            img, mask = random_rotation(img,mask,range_rotate)

        if translation_h_w:    
            img, mask = random_translation(img, mask, translation_h_w)
        
        if zoom_h_w:
            img, mask = random_zoom(img, mask, zoom_h_w)

        return img, mask

    return data_aug


def preprocess_data(data, data_augmentation=False, 
                    return_label_info=False, shape=256, repeat=1,
                    flip_left_right=True, 
                    flip_up_down=True,
                    range_rotate=(-10,10),
                    translation_h_w=None,
                    zoom_h_w=None):

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
                                    range_rotate=range_rotate,
                                    translation_h_w=translation_h_w,
                                    zoom_h_w=zoom_h_w,
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
             range_rotate=(-10,10),
             translation_h_w=None,
             zoom_h_w=None, 
             split = None):
    if split:
        dataset = dataset_class(seed=seed, split=split)
    else: 
        dataset = dataset_class(seed=seed)
    train_data, val_data, test_data = dataset()

    train_data = preprocess_data(
                    train_data, data_augmentation, 
                    return_label_info, shape, repeat=repeat, 
                    flip_left_right=flip_left_right, 
                    flip_up_down=flip_up_down,
                    range_rotate=range_rotate,
                    translation_h_w=translation_h_w,
                    zoom_h_w=zoom_h_w,
                    ).batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

    val_data = preprocess_data(
                                val_data, False, 
                                return_label_info, 
                                shape
                                ).batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

    test_data = preprocess_data(
                                test_data, False, 
                                return_label_info,
                                shape
                                ).batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_data, val_data, test_data