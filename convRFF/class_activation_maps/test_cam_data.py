# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf 

from gcpds.image_segmentation.datasets.segmentation import NerveUtp
from gcpds.image_segmentation.datasets.segmentation import BrachialPlexus
from gcpds.image_segmentation.models import unet_baseline

from tf_keras_vis.layercam import Layercam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from gcpds.image_segmentation.class_activation_maps import SegScore


from cam_data import gen_calculate
from cam_data import save
from cam_data import load

# %%
def resize(shape=(128,128)):
    def func(img,mask):
        return tf.image.resize(img,shape),tf.image.resize(mask,shape)
    return func

dataset = BrachialPlexus()
train, val, test = dataset()


train = train.map(lambda x,y,*l: (*resize()(x,y),*l)).take(50)

# %%
model = unet_baseline(input_shape=(128, 128,1))

# %%
layer_cam = Layercam(model,
                    model_modifier=ReplaceToLinear(),
                    clone=True)

layers = [layer.name for layer in model.layers if 'Conv' in layer.name or 'Line' in layer.name]
targe_classes = [0, 1]
file_path = 'test.mymemmap'

# %%
batch_size = 10
total_rows = len(layers)*len(targe_classes)*len(train)

# %%
total_rows

# %%
generator = gen_calculate(layer_cam, layers, train.batch(batch_size), targe_classes)

# %%
save(generator, total_rows, file_path)


# %%
arr_mmap = load(file_path)

# %%
arr_mmap.dtype

# %%
arr_mmap['info_intance']


