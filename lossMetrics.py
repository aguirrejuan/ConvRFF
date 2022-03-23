from tensorflow.keras import backend as K
import tensorflow as tf 
import numpy as np 

def dice_coef(y_true, y_pred, smooth = 1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def renny_entropy(ytrue,ypred):
    Ke = tf.matmul(ypred,ypred,transpose_b=True)
    Ke = Ke/tf.linalg.trace(Ke)
    #entropy_renny  = (1/(1-2))*tf.math.log((1/tf.constant(32*32,dtype=ypred.dtype))*tf.linalg.trace(tf.matmul(Ke,Ke,transpose_a=True)))
    entropy_renny = tf.constant(1/(1-2))*tf.linalg.trace(tf.matmul(Ke,Ke,transpose_a=True))
    return entropy_renny


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred, smooth = 1.):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true) + K.sum(y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def iou_np(mask,mask_est,smooth=1):
  inter = np.sum(mask*mask_est)
  sum__ = np.sum(mask) + np.sum(mask_est)
  return (inter + smooth)/(sum__ - inter + smooth)


def sensitivity(y_true, y_pred):
    y_true = tf.cast(y_true > 0.5,tf.float32)
    y_pred = tf.cast(y_pred > 0.5 ,tf.float32)
    
    true_positves = K.sum(y_true*y_pred,axis=[1,2,3])
    total_positives = K.sum(y_true,axis=[1,2,3])
   
    return tf.reduce_mean(true_positves / (total_positives + K.epsilon()))

def specificity(y_true, y_pred):
    y_true = tf.cast(y_true < 0.5,tf.float32)
    y_pred = tf.cast(y_pred < 0.5 ,tf.float32)
    return sensitivity(y_true,y_pred)