# ------------------------------------------------------------ #
#
# file : metrics.py
# author : CM
# Metrics for evaluation
#
# ------------------------------------------------------------ #
from keras import backend as K
import tensorflow as tf

# Sensitivity (true positive rate)
def sensitivity(truth, prediction):
	TP = K.sum(K.round(K.clip(truth * prediction, 0, 1)))
	P = K.sum(K.round(K.clip(truth, 0, 1)))
	return TP / (P + K.epsilon())

# Specificity (true negative rate)
def specificity(truth, prediction):
	TN = K.sum(K.round(K.clip((1-truth) * (1-prediction), 0, 1)))
	N = K.sum(K.round(K.clip(1-truth, 0, 1)))
	return TN / (N + K.epsilon())

# Precision (positive prediction value)
def precision(truth, prediction):
	TP = K.sum(K.round(K.clip(truth * prediction, 0, 1)))
	FP = K.sum(K.round(K.clip((1-truth) * prediction, 0, 1)))
	return TP / (TP + FP + K.epsilon())

def dice_coef(y_true, y_pred, smooth=1):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
# def class_balanced_xent(truth, prediction):
# 	sig_prediction = tf.nn.sigmoid(prediction)
# 	#prediction = nn.sigmoid(prediction)
# 	FN = tf.reduce_sum(tf.round(tf.clip_by_value(truth * (1-sig_prediction), 0, 1)))
# 	FP = tf.reduce_sum(tf.round(tf.clip_by_value((1-truth) * sig_prediction, 0, 1)))
# 	TP = tf.reduce_sum(tf.round(tf.clip_by_value(truth * sig_prediction, 0, 1)))
#
# 	xent = tf.reduce_mean(tf.nn.relu(prediction)-prediction*truth+ tf.log(tf.exp(-tf.abs(prediction))+1))
#
# 	gamma1 = 0.5 + (1/tf.reduce_sum(FP))*tf.reduce_sum(tf.abs(FP*(0.5-prediction)))
# 	gamma2 = 0.5 + (1/tf.reduce_sum(FN))*tf.reduce_sum(tf.abs(FN*(sig_prediction-0.5)))
#
# 	L1a = tf.reduce_mean(truth*-tf.log(prediction))
#
# 	L1b = tf.reduce_mean((1-truth)*-tf.log(1-prediction))
#
#
# 	L2a = -(gamma1/tf.reduce_sum(truth))*tf.reduce_sum(FP*tf.log(1-prediction)) ##inf->nan
# 	L2b = -(gamma2/tf.reduce_sum(1-truth))*tf.reduce_sum(FN*tf.log(1+exp(prediction))) ##-100686365.2469 > nan
#
#
#
# 	return xent+L2b#L1a+L1b#+L2b+L2a


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)


def dice_sens_loss(y_true, y_pred, alpha=0.5):
	return -dice_coef(y_true, y_pred) - alpha * sensitivity(y_true, y_pred)

#
# def class_balanced_xent_loss(y_true, y_pred):
# 	return -class_balanced_xent(y_true, y_pred)


def xent(truth, prediction):
	xent_elem_wise = tf.nn.sigmoid_cross_entropy_with_logits(labels=truth, logits=prediction)
	return tf.reduce_mean(xent_elem_wise)