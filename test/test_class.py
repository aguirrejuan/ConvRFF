from tensorflow.keras.applications.vgg16 import VGG16 as Model
import numpy as np
import matplotlib.pyplot as plt


model = Model(weights='imagenet', include_top=True)
model.summary(expand_nested=True)

#========================================================================
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

# Image titles
image_titles = ['Goldfish', 'Bear', 'Assault rifle']

# Load images and Convert them to a Numpy array
img1 = load_img('/home/juan/Documents/tf-keras-vis/examples/images/goldfish.jpg', target_size=(224, 224))
img2 = load_img('/home/juan/Documents/tf-keras-vis/examples/images/bear.jpg', target_size=(224, 224))
img3 = load_img('/home/juan/Documents/tf-keras-vis/examples/images/soldiers.jpg', target_size=(224, 224))
images = np.asarray([np.array(img1), np.array(img2), np.array(img3)])

# Preparing input data for VGG16
X = preprocess_input(images)

## Rendering
#f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
#for i, title in enumerate(image_titles):
#    ax[i].set_title(title, fontsize=16)
#    ax[i].imshow(images[i])
#    ax[i].axis('off')
#plt.tight_layout()
#plt.show()

#==================================================================================
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

replace2linear = ReplaceToLinear()

# Instead of using the ReplaceToLinear instance above,
# you can also define the function from scratch as follows:
def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear


# Instead of using CategoricalScore object,
# you can also define the function from scratch as follows:
def score_function(output):
    # The `output` variable refers to the output of the model,
    # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes)
    results = np.array((output[0][1], output[1][294], output[2][413]))
    return results

from matplotlib import cm
layer = -1 #'block4_conv2'

model.summary()

from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils import num_of_gpus

# Create ScoreCAM object
scorecam = Scorecam(model)#,model_modifier=replace2linear,)

# Generate heatmap with ScoreCAM
cam = scorecam(score_function, X, penultimate_layer='input_1',seek_penultimate_conv_layer=False,
               )

## Since v0.6.0, calling `normalize()` is NOT necessary.
# cam = normalize(cam)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
plt.tight_layout()
plt.show()