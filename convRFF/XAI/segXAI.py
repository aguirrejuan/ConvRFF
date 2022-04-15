import tensorflow as tf 
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.activations import sigmoid

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
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

    def __init__(self,model,data,masks,target_class,layer_name):
        self.target_class  = target_class
        self.masks = masks 
        self.pred_masks = model.predict(data) > 0.5
        self.model = self.__addOutputLayer(model)
        self.data = data 
        self.layer_name = layer_name 
        

    def __addOutputLayer(self,model):
        input = model.input
        
        output_model = model.layers[-1]
        output_model.activation = tf.keras.activations.linear
        output_model = output_model.output

        x = Lambda(_custom_layer(self.target_class),name='Lambda')(output_model)
        output = x#sigmoid(x)
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


    def gradCamPlusPlus(self,):
        gradcam = GradcamPlusPlus(self.model,
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


    def plot(self,cam,nrows=3, ncols=5,figsize=(25, 20)):
        f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        ax = ax.ravel()
        pad_masks = np.zeros_like(self.pred_masks)
        pad_masks[:,1:-1,1:-1,:] = self.pred_masks[:,1:-1,1:-1,:]
        for i, title in enumerate(np.arange(nrows*ncols)):
            heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
            contour = segmentation.clear_border(self.masks[i,...,0])
            pred_contour = segmentation.clear_border(pad_masks[i,...,0])
            ax[i].imshow(self.data[i])
            ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
            ax[i].contour(contour,[0.5],colors=['white'])
            ax[i].contour(pred_contour,[0.5],colors=['red'])
            ax[i].axis('off')
        plt.tight_layout()
        plt.show()


def load_data(number_samples=15):
    from convRFF.models.load_model import load_model
    from convRFF.datasets.nerveUTP import get_data
    TRAIN, VAL, TEST, TOTAL_TRAINING,TOTAL_VALIDATION = get_data(batch_size=number_samples,height=128,width =128)
    for imgs,masks in TEST:
        break
        
    return imgs,masks
    


if __name__ == "__main__":
    from convRFF.models.load_model import load_model
    data,masks = load_data()

    model = load_model()
    segXAI = SegXAI(model, data,masks=masks,target_class=0, layer_name='Trans60')

    #cam = segXAI.gradCam()
    #cam = segXAI.gradCamPlusPlus()
    cam = segXAI.scoreCam()
    

    segXAI.plot(cam)