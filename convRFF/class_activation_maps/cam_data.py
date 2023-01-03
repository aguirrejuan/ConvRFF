import numpy as np 
from tqdm import tqdm
from gcpds.image_segmentation.class_activation_maps import SegScore


# Define data type for memory-mapped file
DTYPE = np.dtype([('info_intance', 'U10', (2,)),  # Unicode string of length 10, shape (2,)
                  ('layer', str, 10),  # String of length 10
                  ('targe_class', int, 1),  # Integer of length 1
                  ('cam', np.float32, (128, 128))  # Float of shape (128, 128)
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
        for img, mask, *info_instance in data:
            # Convert info instances to list of lists of strings
            info_instance = [[i.decode() for i in lists.numpy()] for lists in info_instance]
            info_instance = list(zip(*info_instance))
            for target_class in target_classes:
                # Initialize SegScore object with mask and target class
                seg_score = SegScore(mask,target_class=target_class, logits=True)
                # Generate CAM
                cam = cam_method(seg_score, img, penultimate_layer=layer,
                                 seek_penultimate_conv_layer=False,
                                 normalize_cam=False)
                # Repeat layer and target class for each element in cam
                layer_ = [layer]*len(cam)
                target_class_ = [target_class]*len(cam)
                # Yield tuple of information instances, layers, target classes, and CAMs
                yield info_instance, layer_, target_class_, [c for c in cam]


def save(generator, total_rows, file_path, dtype=DTYPE):
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
        chunk_size = len(data[-1])
        slice_ = slice(i*chunk_size, i*chunk_size + chunk_size)
        filep[slice_] = list(zip(*data))
    # Flush changes to disk
    filep.flush()


def load(file_path, dtype=DTYPE):
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
