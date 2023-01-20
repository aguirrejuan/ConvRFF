import numpy as np 
from tqdm import tqdm
from gcpds.image_segmentation.class_activation_maps import SegScore
from convRFF.data_class import mimic_mmap

# Define data type for memory-mapped file
DTYPE = np.dtype([('info_instance', 'U50', (1188,2)),  
                  ('layer','U50', 1), 
                  ('cam', np.float32, (1188,128, 128, 2))
                 ])


def gen_calculate(cam_method, layers, data, target_classes):
    """
    Generate class activation maps (CAMs) using specified CAM generation method,
    layers, and target classes.

    Parameters:
        cam_method: function for generating CAMs
        layers: list of layers
        data: list of tuples, where each tuple contains an image, a mask, and
            possibly additional information instances
        target_classes: list of target classes

    Yields:
        tuple: information instances, layers, target classes, and CAMs
    """

     
    for layer in layers:
        first_time = True
        cam_per_instance = []
        total_info_instance = []
        for img, mask, *info_instance in data:
            # Convert info instances to list of lists of strings
            info_instance = [[i.decode() for i in lists.numpy()] for lists in info_instance]
            info_instance = list(zip(*info_instance))

            cam_temp_target = []
            for target_class in target_classes:
                # Initialize SegScore object with mask and target class
                seg_score = SegScore(mask,target_class=target_class, logits=True)
                # Generate CAM
                cam = cam_method(seg_score, img, penultimate_layer=layer,
                                 seek_penultimate_conv_layer=False,
                                 normalize_cam=False)
                
                cam_temp_target.append(cam[...,None])
            cam_temp_target = np.concatenate(cam_temp_target, axis=-1)
            if first_time:
                cam_per_instance = cam_temp_target
                first_time = False
            else: 
                cam_per_instance = np.concatenate([cam_per_instance,cam_temp_target], axis=0)
            total_info_instance.extend(info_instance)
        #cam_per_instance = np.concatenate(cam_per_instance, axis=0)
        yield total_info_instance, layer, cam_per_instance


def save_mimic_mmap(generator, file_path, dtype):
    filep = mimic_mmap(file_name, dtype=dtype, mode='w+')
    for data in tqdm(enumerate(generator)):
        infor_instances, layer, infor_instances 
        filep[layer] = infor_instances, infor_instances


def load_mimic_mmap(file_path, dtype):
    filep = mimic_mmap(file_name, dtype=dtype, mode='r')
    return filep


def save_memmap(generator, total_rows, file_path, dtype=DTYPE):
    """
    Save output of generator to memory-mapped file at specified file path.

    Parameters:
        generator: generator object
        total_rows: total number of rows that will be generated
        file_path: file path to save memory-mapped file
        dtype: data type of memory-mapped file
    """
    # Initialize memory-mapped file
    filep = np.memmap(file_path, dtype=dtype, mode='w+', shape=(total_rows,))
    # Iterate over generator, save output to memory-mapped file

    for i, data in tqdm(enumerate(generator)):
        filep[i] = data
    # Flush changes to disk
    filep.flush()


def load_memmap(file_path, dtype=DTYPE):
    """
    Load memory-mapped file from specified file path.

    Parameters:
        file_path: file path of memory-mapped file
        dtype: data type of memory-mapped file

    Returns:
        memory-mapped file
    """
    # Load memory-mapped file
    filep = np.memmap(file_path, dtype=dtype, mode='r')
    return filep
