from tensorflow.keras.models import load_model as ld
from convRFF.models.RFF import RandomFourierFeatures
from convRFF.models.convRFF import ConvRFF
from convRFF.lossMetrics.lossMetrics import dice_coef_loss,iou,dice_coef,sensitivity,specificity


def load_model(path='model.h5'):
    model = ld(path,custom_objects={'RandomFourierFeatures':RandomFourierFeatures,
                                                    'dice_coef_loss':dice_coef_loss,
                                                    'iou':iou,
                                                    'dice_coef':dice_coef,
                                                    'sensitivity':sensitivity,
                                                    'specificity':specificity,
                                                    'ConvRFF':ConvRFF})
    return model 

if __name__ == "__main__":
    import os 
    print(os.getcwd())
    model = load_model('./model.h5')
    model.summary()