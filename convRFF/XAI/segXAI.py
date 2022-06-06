""" Keras vis for segmentation models 
"""

import tensorflow as tf 
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.activations import sigmoid

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.layercam import Layercam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np 
from skimage import segmentation



def _custom_layer(target_class):
    def compute(inputs):
        pred = inputs[0]
        mask = inputs[1]
        if target_class == 1:
            mask = mask
        elif target_class == 0:
            mask = mask-1
        return tf.reduce_mean(mask*pred,axis=[1,2,3])[...,None]
    return compute 


class SegXAI:

    def __init__(self,model,data,masks,target_class,layer_name):
        self.target_class  = target_class
        self.masks = masks 
        self.pred_masks = model.predict(data) > 0.5 #just for plotting 
        self.model = self.__addOutputLayer(model)
        self.data = data 
        self.layer_name = layer_name 
        

    def __addOutputLayer(self,model):
        input = model.input
        input_mask = Input(shape=(None,None,1),name='Input_mask')
        last_layer = model.layers[-1]
        last_layer.activation = tf.keras.activations.linear
        output_model = last_layer.output

        x = Lambda(_custom_layer(self.target_class),name='Lambda')([output_model,input_mask])
        output = x#sigmoid(x)
        model = Model((input,input_mask),output)
        return model


    def score_function(self,pred):
        return pred
        

    def gradCam(self,):
        gradcam = Gradcam(self.model,
                      model_modifier=ReplaceToLinear(),
                      clone=True)

        cam = gradcam(self.score_function,
                  (self.data,self.masks),
                  penultimate_layer=self.layer_name,
                  seek_penultimate_conv_layer=False)
        return cam[0]


    def gradCamPlusPlus(self,):
        gradcam = GradcamPlusPlus(self.model,
                          model_modifier=ReplaceToLinear(),
                          clone=True)
        cam = gradcam(self.score_function, 
                    (self.data,self.masks),
                    penultimate_layer=self.layer_name,
                    seek_penultimate_conv_layer=False)
        return cam[0]


    def scoreCam(self,):
        scorecam = Scorecam(self.model,
                            clone=True,
                            )
        cam = scorecam(self.score_function,
                        (self.data,self.masks),
                        penultimate_layer=self.layer_name,
                        seek_penultimate_conv_layer=False)
        return cam[0]

    def layerCam(self,):
        layercam = Layercam(self.model,
                            model_modifier=ReplaceToLinear(),
                            clone=True)
        cam = layercam(self.score_function,
                        (self.data,self.masks),
                        penultimate_layer=self.layer_name,
                        seek_penultimate_conv_layer=False)
        return cam[0]


    def YcOc(self,cam):
        cam  = cam[...,None]
        Y_c = self.model.predict([self.data,self.masks])
        O_c = self.model.predict([self.data*cam,self.masks])
        return Y_c,O_c


    def average_drop(self,cam):
        Y_c,O_c = self.YcOc(cam)
        return np.sum(np.maximum(0,(Y_c-O_c))/Y_c)*100
    

    def average_increase(self,cam):
        Y_c,O_c = self.YcOc(cam)
        return 100*np.mean(Y_c < O_c)


    def plot(self,cam,nrows=2, ncols=5,figsize=(25, 20)):
        f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        f.suptitle(f"Class :{self.target_class}")

        ax = ax.ravel()
        pad_pred_masks = np.zeros_like(self.pred_masks)
        pad_pred_masks[:,1:-1,1:-1,:] = self.pred_masks[:,1:-1,1:-1,:]
        
        pad_masks = np.zeros_like(self.masks)
        pad_masks[:,1:-1,1:-1,:] = self.masks[:,1:-1,1:-1,:]

        for i, title in enumerate(np.arange(nrows*ncols)):
            heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
            contour = segmentation.clear_border(pad_masks[i,...,0])
            pred_contour = segmentation.clear_border(pad_pred_masks[i,...,0])
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


def load_catDog(num_examples=15):
    from convRFF.datasets.catsDogs import get_data 
    train_batches, test_batches = get_data(num_examples)
    for imgs,masks in test_batches:
        break
    return np.array(imgs),np.array(masks)


if __name__ == "__main__":
    from convRFF.models.load_model import load_model
    data,masks = load_catDog(num_examples=10)

    model = load_model('/home/juan/Downloads/model.h5')
    #model = load_model('/home/juan/Documents/ConvRFF/model.h5')
    #model.summary()
    _class = 1
    segXAI = SegXAI(model, data,masks=masks,target_class=_class, layer_name='conv2d_33') #conv2d_33,conv2d_22

    cam = segXAI.gradCam()
    print(f'Average Drop: {segXAI.average_drop(cam):.3f} \nAverage Increace: {segXAI.average_increase(cam):.3f}')
    segXAI.plot(cam,nrows=2, ncols=5)
    cam = segXAI.gradCamPlusPlus()
    print(f'Average Drop: {segXAI.average_drop(cam):.3f} \nAverage Increace: {segXAI.average_increase(cam):.3f}')
    segXAI.plot(cam,nrows=2, ncols=5)

    cam = segXAI.layerCam()
    
    print(f'Average Drop: {segXAI.average_drop(cam):.3f} \nAverage Increace: {segXAI.average_increase(cam):.3f}')
    segXAI.plot(cam,nrows=2, ncols=5)