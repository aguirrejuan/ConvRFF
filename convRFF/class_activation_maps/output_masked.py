import tensorflow as tf
import numpy as np 

from convRFF.class_activation_maps import save_cam_data
from gcpds.image_segmentation.class_activation_maps import SegScore


DTYPE = np.dtype([('info_instance', 'U50', (2,)),  # Unicode string of length 10, shape (2,)
                  ('layer', 'U50', 1),  # String of length 10
                  ('target_class', int, 1),  # Integer of length 1
                  ('y_pred', np.float32, 1),
                  ('o_pred', np.float32, 1),
                 ])


def generator_dataset(arr_mmap, class_dataset, batch_size=32):
    def gen():
        dataset = class_dataset()
        layers = np.unique(arr_mmap['layer'])
        norm_ter = {layer:arr_mmap[arr_mmap['layer']==layer]['cam'].max() for layer in layers}
        for info_instance, layer, target_class, cam in arr_mmap:
            id_img = info_instance[1]
            img, mask, label, id_image  = dataset.load_instance_by_id(id_img)
            img = tf.image.resize(img,cam.shape)
            mask = tf.image.resize(mask,cam.shape)
            norm_cam = cam/norm_ter[layer]
            img_masked = img*cam[...,None]
            mask = mask if target_class==1 else 1-mask
            yield img, img_masked, mask, info_instance, target_class, layer 


    output_signature = (tf.TensorSpec((None,None,None), tf.float32), 
                    tf.TensorSpec((None,None,None), tf.float32),
                    tf.TensorSpec((None,None,None), tf.float32),
                    tf.TensorSpec(None, tf.string),
                    tf.TensorSpec(None, tf.float32),
                    tf.TensorSpec(None, tf.string))
    
    data = tf.data.Dataset.from_generator(gen,output_signature=output_signature )
    data = data.batch(batch_size)
    
    return data 


def generator_output_model(model, class_data, arr_mmap):
    data  = generator_dataset(arr_mmap, class_data)
    for img, img_masked, mask, info_instance, target_class, layer  in data:

        N = tf.reduce_sum(mask, axis=[1,2,3])
        y_pred = tf.abs(model.predict(img,verbose=0) - (1-target_class))/N
        y_pred = tf.reduce_sum(y_pred*mask, axis=[1,2,3]).numpy()

        o_pred = tf.abs(model.predict(img_masked,verbose=0) - (1-target_class))/N
        o_pred = tf.reduce_sum(o_pred*mask,axis=[1,2,3]).numpy()

        info_instance = [[i.decode() for i in lists.numpy()] for lists in info_instance]
        layer = [l.numpy().decode() for l in layer]
        target_class = target_class.numpy()

        yield info_instance, layer, target_class, y_pred, o_pred 


def save_ouput_model_data(generator, total_rows, file_path, dtype=DTYPE):
    save_cam_data(generator, total_rows, file_path, dtype=dtype)
