import sys
sys.path.append("./")
print(sys.path)

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.layercam import Layercam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from convRFF.models.load_model import load_model
from convRFF.XAI.seg_cam import SegScore
from convRFF.XAI.average_cam import AveragesCam

import tensorflow as tf 


def load_catDog(num_examples=15):
    from convRFF.datasets.catsDogs import get_data 
    train_batches, test_batches = get_data(num_examples)
    for imgs,masks in test_batches:
        break
    return np.array(imgs),np.array(masks)


def plot(cam,data,nrows=2, ncols=5,figsize=(25, 20)):
    f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    ax = ax.ravel()
    for i, title in enumerate(np.arange(nrows*ncols)):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        
        ax[i].imshow(data[i])
        ax[i].imshow(heatmap, cmap='jet', alpha=0.3)
        #ax[i].contour(contour,[0.5],colors=['white'])
        #ax[i].contour(pred_contour,[0.5],colors=['red'])
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

#===========================================================0

data,masks = load_catDog(num_examples=10)
model = load_model('/home/juan/Downloads/model.h5')


score_function = SegScore(masks,class_channel=0)

scorecam = Scorecam(model,
                   clone=True,
                   model_modifier=ReplaceToLinear()
                   )
                   
cam = scorecam(score_function, 
                data,
                penultimate_layer='conv2d_33',
                seek_penultimate_conv_layer=False)
plot(cam,data)

#average = AveragesCam(model,data)
#avg_drop = average.average_drop(cam,score_function)
#avg_increase = average.average_increase(cam,score_function)
#print(avg_drop,avg_increase)