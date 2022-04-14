import tensorflow as tf 

from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model


from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np 
from skimage import segmentation

def _custom_layer(target_class):
    def compute(pred):
        mask = tf.cast(pred > 0.5,tf.float32) if target_class == 1 else -tf.cast(pred <= 0.5,tf.float32) 
        return tf.reduce_mean(mask*pred,axis=[1,2,3])[...,None]
    return compute 

class SegXAI:

    def __init__(self,model,data,target_class,layer_name):
        self.target_class  = target_class
        self.model = self.__addOutputLayer(model)
        self.data = data 
        self.layer_name = layer_name 

    def __addOutputLayer(self,model):
        input = model.input
        
        output_model = model.layers[-1]
        output_model.activation = tf.keras.activations.linear
        output_model = output_model.output

        output = Lambda(_custom_layer(self.target_class),name='Lambda')(output_model)
        model = Model(input,output)
        return model

    def score_function(self,pred):
        return pred
        

    def gradCam(self,):
        gradcam = Gradcam(self.model,
                      model_modifier=ReplaceToLinear(),
                      clone=True)

        cam = gradcam(self.score_function,
                  self.data,
                  penultimate_layer=self.layer_name,
                  seek_penultimate_conv_layer=False)

        return cam 

    def scoreCam(self,):
        scorecam = Scorecam(self.model)
        cam = scorecam(self.score_function,
                        self.data,
                        penultimate_layer=self.layer_name,
                        seek_penultimate_conv_layer=False)

        return cam 


    def plot(self,cam,mask):
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for i, title in enumerate([1,2,3]):
            heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
            contour = segmentation.clear_border(mask[i,...,0])
            ax[i].set_title(title, fontsize=16)
            ax[i].imshow(self.data[i])
            ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
            ax[i].contour(contour,[0.5],colors=['red'])
            ax[i].axis('off')
        plt.tight_layout()
        plt.show()


def load_mode():
    from tensorflow.keras.models import load_model 
    from tensorflow.keras.layers.experimental import RandomFourierFeatures
    from lossMetrics import dice_coef_loss,iou,dice_coef,sensitivity,specificity
    model = load_model('model.h5',custom_objects={'RandomFourierFeatures':RandomFourierFeatures,
                                                    'dice_coef_loss':dice_coef_loss,
                                                    'iou':iou,
                                                    'dice_coef':dice_coef,
                                                    'sensitivity':sensitivity,
                                                    'specificity':specificity})
    return model 


def load_data():
    from datasets.nerveUTP import get_data
    TRAIN, VAL, TEST, TOTAL_TRAINING,TOTAL_VALIDATION = get_data(batch_size=3,height=128,width =128)
    for imgs,masks in TEST:
        break
        
    return imgs,masks
    

if __name__ == "__main__":
    model = load_mode()
    data,masks = load_data()#tf.random.normal(shape=(3,128,128,3))
    #model.summary()
    segXAI = SegXAI(model, data, target_class=1, layer_name='Trans60')
    #cam = segXAI.gradCam()
    cam = segXAI.scoreCam()
    segXAI.plot(cam,masks)