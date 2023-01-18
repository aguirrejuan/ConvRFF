from tensorflow.keras.models import load_model as ld
from convRFF.layers import ConvRFF
from gcpds.image_segmentation.losses import Dicecoefficient
from gcpds.image_segmentation.metrics import (Jaccard, 
                                              Sensitivity,
                                              Specificity,
                                              DicecoefficientMetric
)

def load_model(path='model.h5'):
    model = ld(path, custom_objects={'ConvRFF':ConvRFF, 
                                     'Dicecoefficient':Dicecoefficient,
                                     'Jaccard':Jaccard, 
                                     'Sensitivity':Sensitivity,
                                     'Specificity':Specificity,
                                     'DicecoefficientMetric':DicecoefficientMetric
                                     }
                                     )
    return model 

if __name__ == "__main__":
    import os 
    print(os.getcwd())
    model = load_model('./model.h5')
    model.summary()