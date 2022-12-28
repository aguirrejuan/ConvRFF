from wandb.keras import WandbCallback
from gcpds.image_segmentation.losses import DiceCoeficiente
from gcpds.image_segmentation.metrics import (Jaccard, 
                                              Sensitivity,
                                              Specificity,
                                              DiceCoeficienteMetric
)

from convRFF.data import get_data

def get_compile_parameters():
  return {'loss':DiceCoeficiente(),
          'optimizer':tf.keras.optimizers.Adam(learning_rate=1e-3),
          'metrics':[Jaccard(), 
                     Sensitivity(), 
                     Specificity(),
                     DiceCoeficienteMetric(),
                     'binary_accuracy']
  }


def get_train_parameters(dataset_class):
    train_data, val_data, test_data = get_data(dataset_class)
    return {'x':train_data,
            'validation_data':val_data,
            'epochs':200,
            'callbacks':[WandbCallback(save_model=True)]
    }


def train(model,dataset_class):
    model.compile(**get_compile_parameters())
    model.fit(**get_train_parameters(dataset_class))