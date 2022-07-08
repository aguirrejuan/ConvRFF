""" Keras vis for clasification models 
"""
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.layercam import Layercam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from tensorflow.keras.backend import epsilon



class Cams:

    def __init__(self,model,images,labels,layer):
        self.model = model
        self.images = images 
        self.labels = labels 
        self.layer = layer 
        self.cams = {}

    def score_function(self,output):
        return [output[i,label] for i,label in enumerate(self.labels)]


    def gradCam(self,):
        gradcam = Gradcam(self.model,
                      model_modifier=ReplaceToLinear(),
                      clone=True)

        cams = gradcam(self.score_function,
                  self.images,
                  penultimate_layer=self.layer,
                  seek_penultimate_conv_layer=False)

        self.cams['gradcam'] = cams


    def gradCamPlusPlus(self,):
        gradcam = GradcamPlusPlus(self.model,
                      model_modifier=ReplaceToLinear(),
                      clone=True)

        cams = gradcam(self.score_function,
                  self.images,
                  penultimate_layer=self.layer,
                  seek_penultimate_conv_layer=False)
        
        self.cams['gradcamplusplus'] = cams


    def scoreCam(self,):
        scorecam = Scorecam(self.model,
                            clone=True)
        cams = scorecam(self.score_function,
                             self.images,
                             penultimate_layer=self.layer,
                                seek_penultimate_conv_layer=False,)

        self.cams['scorecam'] = cams 
        
    
    def layerCam(self,):
        layercam = Layercam(self.model,
                            model_modifier=ReplaceToLinear(),
                            clone=True)
        cams = layercam(self.score_function,
                        self.images,
                        penultimate_layer=self.layer,
                        seek_penultimate_conv_layer=False)

        self.cams['layercam'] = cams 


    def run_cams(self,):
        self.gradCam()
        self.gradCamPlusPlus()
        self.scoreCam()
        self.layerCam()


    def _YcOc(self,cam,filter_correct_labels,return_oc=False):
        cam  = cam[...,None]
        Y_c = self.model.predict(self.images)[np.arange(len(cam)), self.labels]
        scores_o_c = self.model.predict(self.images*cam)
        O_c = scores_o_c[np.arange(len(cam)), self.labels]
        
        if filter_correct_labels:
            mask = np.argmax(scores_o_c,axis=-1) == self.labels
            Y_c = Y_c[mask]
            O_c = O_c[mask]
        
        if not return_oc:
            return Y_c,O_c
        else: 
            return (Y_c,O_c),scores_o_c


    def _average_drop(self,cam,filter_correct_labels,return_oc=False):
        if not return_oc:
            Y_c,O_c = self._YcOc(cam,filter_correct_labels)
            return 100*np.maximum(0,(Y_c-O_c))/(Y_c+epsilon())
        else:
            Y_c,O_c,score_oc = self._YcOc(cam,filter_correct_labels,return_oc)
            return 100*np.maximum(0,(Y_c-O_c))/(Y_c+epsilon()),score_oc


    def _average_increase(self,cam,filter_correct_labels):
        Y_c,O_c = self._YcOc(cam,filter_correct_labels)
        return 100*np.mean(Y_c < O_c)

    def _average_relative_increase(self,cam,filter_correct_labels,return_oc=False):
        if not return_oc:
            Y_c,O_c = self._YcOc(cam,filter_correct_labels)
            return 100*(O_c-Y_c)/(Y_c+epsilon())
        else:
            Y_c,O_c,score_oc = self._YcOc(cam,filter_correct_labels,return_oc)
            return 100*(O_c-Y_c)/(Y_c+epsilon()),score_oc

    def averages_drops_vector(self,filter_correct_labels=False,return_oc=False):
        return {name:self._average_drop(cams,filter_correct_labels,return_oc) for name,cams in self.cams.items()}

    def averages_drops(self,filter_correct_labels=False):
        return {name:np.mean(self._average_drop(cams,filter_correct_labels)) for name,cams in self.cams.items()}

    def averages_increases(self,filter_correct_labels=False):
        return {name:self._average_increase(cams,filter_correct_labels) for name,cams in self.cams.items()}

    def averages_relative_increases(self,filter_correct_labels=False):
        return {name:np.mean(self._average_relative_increase(cams,filter_correct_labels)) for name,cams in self.cams.items()}
    
    def averages_relative_increases_vector(self,filter_correct_labels=False,return_oc=False):
        return {name:self._average_relative_increase(cams,filter_correct_labels,return_oc) for name,cams in self.cams.items()}

    def get_list_results(self):
        results = {}
        for name, cam in self.cams.items():
            cams = [np.uint8(cm.jet(self.cams[name][i])[..., :3] * 255) for i in range(len(self.cams[name]))]
            images = np.squeeze(self.images)
            images = [images[i] for i in range(len(images))]
            results[name] = {'labels':self.labels,'cams':cams,'images':images}
        return results


    def get_average_cams_per_class(self,)-> np.array:
        results = {}
        for name, cam in self.cams.items():
            results[name] =  [np.mean(self.cams[name][self.labels == i],axis=0) for i in np.sort(np.unique(self.labels))]
        return results