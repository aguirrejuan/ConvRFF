import tensorflow as tf 
import tensorflow_datasets as tfds 


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask != 2,tf.float32)
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def get_data(batch_size=16):
    config_tfds = tfds.ReadConfig(shuffle_seed=42)
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True,read_config=config_tfds)

    train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = batch_size
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    train_batches = (train_images
        .cache()
        .shuffle(BUFFER_SIZE,seed=42)
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(BATCH_SIZE)
    return train_batches, test_batches


if __name__ == "__main__":
    train_batches, test_batches = get_data()
    for x,y in train_batches.take(1):
        print(x.shape)
        