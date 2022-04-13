import tensorflow as tf 

from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np 

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


    def plot(self,cam):
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for i, title in enumerate([1,2,3]):
            heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
            ax[i].set_title(title, fontsize=16)
            ax[i].imshow(self.data[i])
            ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
            ax[i].axis('off')
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    from models.unetConvRFF import get_model 
    data = tf.random.normal(shape=(3,128,128,3))
    model = get_model()
    #model.summary()
    segXAI = SegXAI(model, data, target_class=1, layer_name='ConvRFF')
    cam = segXAI.gradCam()
    #cam = segXAI.scoreCam()
    segXAI.plot(cam)