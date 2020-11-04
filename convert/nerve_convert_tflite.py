from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import cv2
import csv
import re
from glob import glob as glb

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import model.nerve_net as nerve_net 
import input.nerve_input as nerve_input
#from run_length_encoding import RLenc
from utils.experiment_manager import make_checkpoint_path
#import utils.metric as metc

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 使用第二块GPU（从0开始）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"       # 使用第二块GPU（从0开始）


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', '../checkpoints',
                            """dir to store trained net """)
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('max_steps', 603780, # 417390, #227980,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_float('keep_prob', 0.668,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('learning_rate', 1e-5,
                            """ keep probability for dropout """)
#tf.app.flags.DEFINE_bool('view_images', 'False',
#                            """ If you want to view image and generated masks""")

TEST_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)


def convert():
  converter = tf.contrib.lite.toco_convert.from_saved_model(saved_model_dir)
  tflite_model = converter.convert()
  open("converted_model.tflite", "wb").write(tflite_model)

def evaluate():
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """

  with tf.Graph().as_default():
    # Make image placeholder
    images_i = tf.placeholder(tf.float32, [1, 210, 290, 1],name="input")

    # Build a Graph that computes the logits predictions from the
    # inference model.
    images_o = nerve_net.inference(images_i,1.0) 
    #metric = nerve_net.metric_image_score(mask,images_gt)

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    
    sess = tf.Session()
    # Use CPU instead GPU
    #sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))

    ckpt = tf.train.get_checkpoint_state(TEST_DIR)

    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    #summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
    #                                        graph_def=graph_def)
    
    tflite_model = tf.contrib.lite.toco_convert(sess.graph_def,[images_i],[images_o])
    open("converted_model.tflite", "wb").write(tflite_model) 
    
    # calc logits 
    # generated_mask = sess.run([mask],feed_dict={images_op: img})
    

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()
  #convert()

if __name__ == '__main__':
  tf.app.run()
