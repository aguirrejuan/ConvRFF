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


    def _YcOc(self,cam):
        cam  = cam[...,None]
        Y_c = self.model.predict(self.images)[np.arange(len(cam)),self.labels]
        O_c = self.model.predict(self.images*cam)[np.arange(len(cam)),self.labels]
        return Y_c,O_c


    def _average_drop(self,cam):
        Y_c,O_c = self._YcOc(cam)
        return np.mean(np.maximum(0,(Y_c-O_c))/Y_c)*100


    def _average_increase(self,cam):
        Y_c,O_c = self._YcOc(cam)
        return 100*np.mean(Y_c < O_c)


    def averages_drops(self,):
        return {name:self._average_drop(cams) for name,cams in self.cams.items()}


    def averages_increases(self,):
        return {name:self._average_increase(cams) for name,cams in self.cams.items()}


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

