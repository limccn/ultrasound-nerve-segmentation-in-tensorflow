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
from run_length_encoding import RLenc
from utils.experiment_manager import make_checkpoint_path
import utils.metric as metc

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 使用第二块GPU（从0开始）
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"       # 使用第二块GPU（从0开始）


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', '../checkpoints',
                            """dir to store trained net """)
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('max_steps', 21000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_float('keep_prob', 0.668,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('learning_rate', 1e-5,
                            """ keep probability for dropout """)
#tf.app.flags.DEFINE_bool('view_images', 'False',
#                            """ If you want to view image and generated masks""")

TEST_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)

def tryint(s):
  try:
    return int(s)
  except:
    return s

def alphanum_key(s):
  return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def metric_image_score(prediction, mask):
  #dice
  intersection = np.sum(prediction * mask)
  dice = (2. * intersection + 1.) / (np.sum(mask) + np.sum(prediction) + 1.)
  loss = 1-dice
  #VOE
  overload_error = np.sum(prediction - mask)
  voe = (2. * overload_error + 1.) / (np.sum(mask) + np.sum(prediction) + 1.)
  #basic
  tp = np.sum(prediction * mask)
  tn = np.sum((1-prediction) * (1-mask))
  fp = np.sum(prediction * (1-mask))
  fn = np.sum((1-prediction) * mask)
  # score
  jaccard = (tp + 1.) / (tp + fn + fp + 1.)
  f1_score = (2. * tp + 1.)/(2. * tp + fn + fp + 1.)
  f2_score = (5. * tp + 1.)/(5. * tp + 4. * fn + fp + 1.)
  tpr = 0 if tp==0 else (tp + 1.) / (tp + fn + 1.)
  fpr = 0 if fp==0 else (fp + 1.) / (fp + tn + 1.)
  tnr = 0 if tn==0 else (tn + 1.) / (fp + tn + 1.)
  fnr = 0 if fn==0 else (fn + 1.) / (tp + fn + 1.)
  precision = (tp + 1.) / (tp + fp + 1.)
  accuracy = (tp + tn + 1.) / (tp + tn + fp + fn + 1.)
  bacc = (tpr + tnr)/ 2.

  return [tp,tn,fp,fn,tpr,tnr,fpr,fnr,accuracy,precision,loss,dice,jaccard,f1_score,f2_score,bacc]


def metric_image_distance(prediction, mask):
  if 0 == np.count_nonzero(prediction) \
    or 0 == np.count_nonzero(mask):
    return ['','','','']
    
  asd = metc.asd(prediction,mask)
  #obj_asd = metc.obj_asd(prediction,mask)
  msd = metc.msd(prediction,prediction)
  assd = metc.assd(prediction,mask)
  hd = metc.hd(prediction,mask)
  return [asd,msd,assd,hd]


def metric_min_area_rect_circle(prediction, mask):
  if 0 == np.count_nonzero(prediction) \
    or 0 == np.count_nonzero(mask):
    return [0,0,0,0,0]

  #图像相交
  img_and = prediction * mask
  img_and = np.uint8(img_and * 255)

  img_and_canny = cv2.Canny(img_and, 100, 300)
  ret, thresh = cv2.threshold(img_and_canny, 127, 255, cv2.THRESH_BINARY)
  img_and_canny = thresh

  # binary是最后返回的二值图像
  #findContours()第一个参数是源图像、第二个参数是轮廓检索模式，第三个参数是轮廓逼近方法
  #输出是轮廓和层次结构，轮廓是图像中所有轮廓的python列表，每个单独的轮廓是对象边界点的(x,y)坐标的Numpy数组
  binary, contours, hierarchy = cv2.findContours(img_and_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  #加粗
  img_and_canny = cv2.drawContours(img_and_canny, contours, -1, (255,255,255), 4)
  binary, contours, hierarchy = cv2.findContours(img_and_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  x=0
  y=0
  w=0
  h=0
  r=0
  for c in contours:
    if cv2.contourArea(c) < 10:
        continue
    #x, y, w, h = cv2.boundingRect(c)
    #cv2.rectangle(img_ori, (x,y), (x+w,y+h), (0,255,0), 1)
    box = cv2.minAreaRect(c)
    x, y = box[0]
    w, h = box[1]
    d = w if w < h else h
    r = int(d/2)
    # find first only
    break
  return [x,y,w,h,r]


def find_contours(img):
  img = np.uint8(img*255.)
  img_canny = cv2.Canny(img, 100, 300)
  ret, thresh = cv2.threshold(img_canny, 127, 255, cv2.THRESH_BINARY)
  img_canny = thresh

  # binary是最后返回的二值图像
  #findContours()第一个参数是源图像、第二个参数是轮廓检索模式，第三个参数是轮廓逼近方法
  #输出是轮廓和层次结构，轮廓是图像中所有轮廓的python列表，每个单独的轮廓是对象边界点的(x,y)坐标的Numpy数组
  binary, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return contours

def evaluate():
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  # get a list of image filenames
  filenames = glb('../data/score/*')
  # sort the file names but this is probably not ness
  #filenames.sort(key=alphanum_key)
  #num_files = len(filename)

  test_filename = [s for s in filenames if "mask" not in s]
  truth_filename = [s for s in filenames if "mask" in s]

  test_filename.sort(key=alphanum_key)
  truth_filename.sort(key=alphanum_key)

  pair_filename = []

  for image in test_filename:
    key = image[:-4]
    truth = [s for s in truth_filename if key+'_mask' in s][0] 
    pair_filename.append((image, truth))
  
  #print(pair_filename)
  print ("file_count:%d"%len(pair_filename))


  with tf.Graph().as_default():
    # Make image placeholder
    images_op = tf.placeholder(tf.float32, [1, 210, 290, 1])
    images_gt = tf.placeholder(tf.float32, [1, 210, 290, 1])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    mask = nerve_net.inference(images_op,1.0) 
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

    # make csv file
    #csvfile = open('test.csv', 'wb') 
    csvfile = open('test_huashan.score.csv', 'w') 
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['img','truth','tp','tn','fp','fn','tpr','tnr','fpr','fnr','acc','prec','loss','dice','jaccard','f1','f2','bacc','asd','msd','assd','hd','center_x','center_y','rect_w','rect_h','radius'])

    for (f,t) in pair_filename:
      # name to save 
      prediction_path = '../data/prediction/'
      name = f[13:-4]
      name_truth=t[13:-4]
      print(name)
     
      # read in image
      img_origin = cv2.imread(f,cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img_origin.copy(), cv2.COLOR_BGR2GRAY)
      # resize
      img = cv2.resize(img,(290,210),interpolation=cv2.INTER_CUBIC)
      
      img = img - np.mean(img)
 
      # format image for network
      img = np.expand_dims(img, axis=0)
      img = np.expand_dims(img, axis=3)

      img_gt = cv2.imread(t, 0)
      img_gt = cv2.resize(img_gt,(290,210),interpolation=cv2.INTER_CUBIC)

      # calc logits 
      generated_mask = sess.run([mask],feed_dict={images_op: img})
    
      #generated_mask = cv2.resize(generated_mask,(580,410))
      generated_mask = generated_mask[0]
      generated_mask = generated_mask[0, :, :, :]
     
      generated_mask = cv2.resize(generated_mask,(580,420),interpolation=cv2.INTER_CUBIC)
      ground_truth_mask = cv2.resize(img_gt,(580,420),interpolation=cv2.INTER_CUBIC)


      # bin for converting to row format
      threshold = .5
      generated_mask[:][generated_mask[:]<=threshold]=0 
      generated_mask[:][generated_mask[:]>threshold]=1

      ground_truth_mask[:][ground_truth_mask[:]<=threshold]=0 
      ground_truth_mask[:][ground_truth_mask[:]>threshold]=1
      
      row = []
      row.append(name)
      row.append(name_truth)
      row.extend(metric_image_score(generated_mask,ground_truth_mask))
      row.extend(metric_image_distance(generated_mask,ground_truth_mask))
      
      overlay_param = metric_min_area_rect_circle(generated_mask,ground_truth_mask)
      print(overlay_param)
      row.extend(overlay_param)

      writer.writerow(row)

      #run_length_encoding = RLenc(generated_mask)
      #print(run_length_encoding)
      #name = name.encode('utf8')
      #run_length_encoding = run_length_encoding.encode('utf8')
      #writer.writerow([name, run_length_encoding])


      '''
      # convert to display 
      generated_mask = np.uint8(generated_mask * 255)
 
      # display image
      cv2.imshow('img', img[0,:,:,0])
      cv2.waitKey(0)
      cv2.imshow('mask', generated_mask[:,:,0])
      cv2.waitKey(0)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      '''
    
      
      save_prediction_path = '../data/prediction_save/'
      filepath_pred = "%s%s.pred.%s"%(save_prediction_path,name,f[-3:])
      filepath_mask = "%s%s.pred.mask.%s"%(save_prediction_path,name,f[-3:])
      #filepath_overlap = "%s%s.overlap.mask.%s"%(save_prediction_path,name,f[-3:])
      filepath_overlap = "%s%s.overlap.mask.%s"%(save_prediction_path,name,'png')
      
      x = int(overlay_param[0])
      y = int(overlay_param[1])
      #r = 30 # 20px
      r = int(overlay_param[4])
      overlap = img_origin.copy()
      contours = find_contours(ground_truth_mask)
      #overlay
      overlap = cv2.drawContours(overlap, contours, -1, (255, 0 , 0), 1)

      #if r > 10 :
      
      cv2.circle(overlap, (x,y), r, (0, 0, 255), 1)
      cv2.circle(overlap, (x,y), 30, (0, 255, 0), 1) 
      cv2.line(overlap, (x-r,y),(x+r,y), (0, 0, 255), 1)
      cv2.line(overlap, (x,y-r),(x,y+r), (0, 0, 255), 1)

      # convert to display 
      #generated_mask = np.uint8(generated_mask * 255)

      overlap = cv2.resize(overlap,(580,420),interpolation=cv2.INTER_CUBIC)

      #cv2.imwrite(filepath_pred, img)
      #cv2.imwrite(filepath_mask, generated_mask[:,:,0])
      cv2.imwrite(filepath_overlap, overlap)
      

      generated_mask = np.uint8(generated_mask)

      if False: 
        # display image
        cv2.imshow('img', np.uint8(img[0,:,:,0]*255.0))
        cv2.waitKey(0)
        cv2.imshow('mask', generated_mask[:,:,0]*255)
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
