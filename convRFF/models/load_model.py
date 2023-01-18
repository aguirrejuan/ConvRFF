from tensorflow.keras.models import load_model as ld
from convRFF.layers import ConvRFF
from gcpds.image_segmentation.losses import DiceCoefficient
from gcpds.image_segmentation.metrics import (Jaccard, 
                                              Sensitivity,
                                              Specificity,
                                              DiceCoefficientMetric
)

def load_model(path='model.h5'):
    model = ld(path, custom_objects={'ConvRFF':ConvRFF, 
                                     'DiceCoefficient':DiceCoefficient,
                                     'Jaccard':Jaccard, 
                                     'Sensitivity':Sensitivity,
                                     'Specificity':Specificity,
                                     'DiceCoefficientMetric':DiceCoefficientMetric
                                     }
                                     )
    return model 

if __name__ == "__main__":
    import os 
    print(os.getcwd())
    model = load_model('./model.h5')
    model.summary()