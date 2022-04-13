import tensorflow as tf 

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear


class SegXAI:

    def __init__(self,model,data,target_class,layer_name):
        self.model = model
        self.data = data 
        self.target_class  = target_class
        self.layer_name = layer_name 


    def score_function(self,pred):
        mask = tf.cast(pred > 0.5,tf.float32) if self.target_class == 1 else -tf.cast(pred <= 0.5,tf.float32) 
        return tf.reduce_mean(mask*pred,axis=[1,2,3])
        

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
        ...




if __name__ == "__main__":
    from models.unetConvRFF import get_model 
    data = tf.random.normal(shape=(5,128,128,3))
    model = get_model()
    model.summary()
    segXAI = SegXAI(model, data, target_class=1, layer_name='ConvRFF')
    #cam = segXAI.gradCam()
    cam = segXAI.scoreCam()
    print(cam.shape)