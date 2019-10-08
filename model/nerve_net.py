
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import model.nerve_architecture
import input.nerve_input as nerve_input
import utils.metric as metric

FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
tf.app.flags.DEFINE_string('model', 'ced',
                           """ model name to train """)
tf.app.flags.DEFINE_string('output_type', 'mask_image',
                           """ What kind of output, possibly image. Maybe other in future """)
tf.app.flags.DEFINE_integer('nr_res_blocks', 1,
                           """ nr res blocks """)
tf.app.flags.DEFINE_bool('gated_res', True,
                           """ gated resnet or not """)
tf.app.flags.DEFINE_string('nonlinearity', 'concat_elu',
                           """ nonlinearity used such as concat_elu, elu, concat_relu, relu """)

def inputs(batch_size):
  """makes input vector
  Return:
    x: input vector, may be filled 
  """
  x, mask = nerve_input.nerve_inputs(batch_size)
  return x, mask

def inference(inputs, keep_prob):
  """Builds network.
  Args:
    inputs: input to network 
    keep_prob: dropout layer
  """
  if FLAGS.model == "ced": 
    prediction = model.nerve_architecture.conv_ced(inputs, nr_res_blocks=FLAGS.nr_res_blocks, keep_prob=keep_prob, nonlinearity_name=FLAGS.nonlinearity, gated=FLAGS.gated_res)

  return prediction 

def loss_image(prediction, mask):
  """Calc loss for predition on image of mask.
  Args.
    inputs: prediction image 
    mask: true image 

  Return:
    error: loss value
  """
  print(prediction.get_shape())
  print(mask.get_shape())
  #mask = tf.flatten(mask)
  #prediction = tf.flatten(prediction)
  intersection = tf.reduce_sum(prediction * mask)
  loss = -(2. * intersection + 1.) / (tf.reduce_sum(mask) + tf.reduce_sum(prediction) + 1.)
  tf.summary.scalar('loss', loss)

  return loss

def metric_image(prediction, mask):
  #mask = tf.flatten(mask)
  #prediction = tf.flatten(prediction)
  intersection = tf.reduce_sum(prediction * mask)
  dice = (2. * intersection + 1.) / (tf.reduce_sum(mask) + tf.reduce_sum(prediction) + 1.)
  tf.summary.scalar('Dice', dice)

  #VOE
  overload_error = tf.reduce_sum(prediction - mask)
  voe = (2. * overload_error + 1.) / (tf.reduce_sum(mask) + tf.reduce_sum(prediction) + 1.)
  tf.summary.scalar('VOE=Volumetric Overlap Error', voe)

  #conformity coefficient
  #iconf_loss=(3.*loss - 2.) / loss
  #tf.summary.scalar('conformity', conf_loss)

  tp = tf.reduce_sum(prediction * mask)
  tn = tf.reduce_sum((1-prediction) * (1-mask))
  fp = tf.reduce_sum(prediction * (1-mask))
  fn = tf.reduce_sum((1-prediction) * mask)

  jaccard = (tp + 1.) / (tp + fn + fp + 1.)
  tf.summary.scalar('IoU Socre=Jaccard Coff', jaccard)

  f1_score = (2. * tp + 1.)/(2. * tp + fn + fp + 1.)
  tf.summary.scalar('F1 Score=Dice Coff', f1_score)

  f2_score = (5. * tp + 1.)/(5. * tp + 4. * fn + fp + 1.)
  tf.summary.scalar('F2 Score', f2_score)

  sensitivity = (tp + 1.) / (tp + fn + 1.)
  tf.summary.scalar('Recall=Sensitivity=TPR', sensitivity)

  fpr = (fp + 1.) / (fp + tn + 1.)
  tf.summary.scalar('FPR', fpr)

  specificity = (tn + 1.) / (fp + tn + 1.)
  tf.summary.scalar('Specificity=TNR', specificity)
  
  fnr =  (fn + 1.) / (tp + fn + 1.)
  tf.summary.scalar('FNR', fnr)

  precision = (tp + 1.) / (tp + fp + 1.)
  tf.summary.scalar('Precision', precision)

  accuracy = (tp + tn + 1.) / (tp + tn + fp + fn + 1.)
  tf.summary.scalar('Accuracy', accuracy)
  #recall = (tp + 1.) / (tp + fn + 1.)
  #tf.summary.scalar('Recall', recall)

  #Precision
  pixel_precision = (intersection + 1.) / (tf.reduce_sum(prediction) + 1.)
  tf.summary.scalar('Pixel Precision', pixel_precision)

  #Recall
  pixel_recall= (intersection + 1.) / (tf.reduce_sum(mask) + 1.)
  tf.summary.scalar('Pixel Recall', pixel_recall)
  
  #BACC
  bacc = (specificity + sensitivity)/ 2.
  tf.summary.scalar('BACC', bacc)
  
  metric_dict = dict()
  metric_dict["tp"]=tp
  metric_dict["tn"]=tn
  metric_dict["fp"]=fp
  metric_dict["fn"]=fn
  metric_dict["fpr"]=fpr
  metric_dict["tpr"]=sensitivity
  metric_dict["tnr"]=specificity
  metric_dict["fnr"]=fnr
  metric_dict["acc"]=accuracy
  metric_dict["dice"]=dice
  metric_dict["iou"]=jaccard
  metric_dict["f1"]=f1_score
  metric_dict["f2"]=f2_score
  metric_dict["bacc"]=bacc

  return metric_dict

def metric_distance(prediction, mask):
  pass

def train(total_loss, lr):
   train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
   return train_op

