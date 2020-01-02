
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
    prediction = model.nerve_architecture.conv_ced(inputs, nr_res_blocks=FLAGS.nr_res_blocks, keep_prob=keep_prob, nonlinearity_name=FLAGS.nonlinearity, gated=FLAGS.gated_res, is_train=False)

    #prediction_sigmod = tf.sigmoid(prediction)
    #tf.summary.image('predicted', prediction_sigmod)

    return prediction


def cross_entropy_loss(prediction, mask):
  print("d",prediction.get_shape())
  print("l",mask.get_shape())

  #y = mask
  #y_pred = prediction
  #bce_loss=-y * (np.log(y_pred)) - (1 - y) * np.log(1 - y_pred)
  #bce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=mask, logits=prediction)
  #bce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=mask, logits=prediction)
  #bce_loss = prediction*tf.log(tf.clip_by_value(,1e-10,1.0)))
  
  #y = prediction
  #y = tf.reshape(prediction, [-1, 1])
  #y_ = mask
  #y_ = tf.reshape(mask, [-1, 1])

  #print("dy",y.get_shape())
  #print("ly",y_.get_shape())

  #y=tf.nn.softmax(y)
  #bce_loss1 = -y_ * tf.log(y) - (1 - y_) * tf.log(1 - y)

  bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=prediction)
  
  #loss = tf.reduce_mean(bce_loss)  
  #bce  = tf.reduce_sum(bce_loss)  
  #print("loss%s"%loss)

  #tf.summary.scalar('sum_y', tf.reduce_sum(y))
  #tf.summary.scalar('sum_y_', tf.reduce_sum(y_))
  #tf.summary.scalar('bce1', tf.reduce_sum(bce_loss1))
  tf.summary.scalar('bce', tf.reduce_sum(bce_loss))
  #tf.summary.scalar('bce_loss1', tf.reduce_mean(bce_loss1))
  tf.summary.scalar('bce_loss', tf.reduce_mean(bce_loss))
  
  return tf.reduce_mean(bce_loss)

def dice_loss(prediction, mask):
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
  tf.summary.scalar('soft_dice_loss', loss)

  return loss

def loss_image(prediction, mask):
  #bce = cross_entropy_loss(prediction, mask)
  dice = dice_loss(prediction, mask)
  #bce_dice = (bce+dice)/2.0
  
  #tf.summary.scalar('bce_dice', bce_dice)

  return dice

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

  jaccard = (tp + 0.0001) / (tp + fn + fp + 0.0001)
  tf.summary.scalar('IoU Socre=Jaccard Coff', jaccard)

  f1_score = (2. * tp + 0.0001)/(2. * tp + fn + fp + 0.0001)
  tf.summary.scalar('F1 Score=Dice Coff', f1_score)

  f2_score = (5. * tp + 0.0001)/(5. * tp + 4. * fn + fp + 0.0001)
  tf.summary.scalar('F2 Score', f2_score)

  sensitivity = (tp + 0.0001) / (tp + fn + 0.0001)
  tf.summary.scalar('Recall=Sensitivity=TPR', sensitivity)

  fpr = (fp + 0.0001) / (fp + tn + 0.0001)
  tf.summary.scalar('FPR', fpr)

  specificity = (tn + 0.0001) / (fp + tn + 0.0001)
  tf.summary.scalar('Specificity=TNR', specificity)
  
  fnr = (fn + 0.0001) / (tp + fn + 0.0001)
  tf.summary.scalar('FNR', fnr)

  precision = (tp + 0.0001) / (tp + fp + 0.0001)
  tf.summary.scalar('Precision', precision)

  accuracy = (tp + tn + 0.0001) / (tp + tn + fp + fn + 0.0001)
  tf.summary.scalar('Accuracy', accuracy)
  #recall = (tp + 1.) / (tp + fn + 1.)
  #tf.summary.scalar('Recall', recall)

  #Precision
  pixel_precision = (intersection + 0.0001) / (tf.reduce_sum(prediction) + 0.0001)
  tf.summary.scalar('Pixel Precision', pixel_precision)

  #Recall
  pixel_recall= (intersection + 0.0001) / (tf.reduce_sum(mask) + 0.0001)
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

