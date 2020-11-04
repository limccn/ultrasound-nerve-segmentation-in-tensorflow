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
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 使用第二块GPU（从0开始）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"       # 使用第二块GPU（从0开始）


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', '../checkpoints',
                            """dir to store trained net """)
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('max_steps', 1144490, #238890, #1028690, #767791,  #603780, # 417390, #227980,
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




def metric_img_mar_circle(image):
  if 0 == np.count_nonzero(image): 
    return [0,0,0,0,0]

  #图像相交
  image = np.uint8(image * 255)

  img_canny = cv2.Canny(image, 100, 300)
  ret, thresh = cv2.threshold(img_canny, 127, 255, cv2.THRESH_BINARY)
  img_canny = thresh

  # binary是最后返回的二值图像
  #findContours()第一个参数是源图像、第二个参数是轮廓检索模式，第三个参数是轮廓逼近方法
  #输出是轮廓和层次结构，轮廓是图像中所有轮廓的python列表，每个单独的轮廓是对象边界点的(x,y)坐标的Numpy数组
  binary, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  #加粗
  img_canny = cv2.drawContours(img_canny, contours, -1, (255,255,255), 4)
  binary, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  x=0
  y=0
  w=0
  h=0
  r=0
  d=0
  for c in contours:
    if cv2.contourArea(c) < 10:
        continue
    #x, y, w, h = cv2.boundingRect(c)
    #cv2.rectangle(img_ori, (x,y), (x+w,y+h), (0,255,0), 1)
    box = cv2.minAreaRect(c)
    t_w, t_h = box[1]
    t_d = t_w if t_w < t_h else t_h
    # get
    if t_d > d: 
      x, y = box[0]
      w, h = box[1]
      d = w if w < h else h
      r = int(d/2)
      # find first only
    
  return [x,y,w,h,r]


def metric_bounding_rect_iou(prediction, mask):
  if 0 == np.count_nonzero(prediction) \
    or 0 == np.count_nonzero(mask):
    return [0,0,0,0]

  x1=0
  y1=0
  w1=0
  h1=0
  area_pred = 0
  contour_pred = find_contours(prediction)
  for c in contour_pred:
    if cv2.contourArea(c) < 10:
        continue
    x1,y1,w1,h1 = cv2.boundingRect(c)
    area_pred = w1*h1
    break

  area_mask = 0
  x2=0
  y2=0
  w2=0
  h2=0
  contour_mask = find_contours(mask)
  for c in contour_mask:
    if cv2.contourArea(c) < 10:
        continue
    x2,y2,w2,h2 = cv2.boundingRect(c)
    area_mask = w2*h2
    break

  if area_mask > 0 and area_pred > 0:
    if(x1>x2+w2):
        iou=0
    if(y1>y2+h2):
        iou=0
    if(x1+w1<x2):
        iou=0
    if(y1+h1<y2):
        iou=0
    colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    iou = overlap_area / (area1 + area2 - overlap_area)
    return [overlap_area,area1,area2,iou]
  return [0,area_pred,area_mask,0]

def metric_min_area_rect_iou(prediction, mask):
  if 0 == np.count_nonzero(prediction) \
    or 0 == np.count_nonzero(mask):
    return [0,0,0,0]

  area_pred = 0
  contour_pred = find_contours(prediction)
  for c in contour_pred:
    if cv2.contourArea(c) < 10:
        continue
    box_pred = cv2.minAreaRect(c)
    points = cv2.boxPoints(box_pred)
    points = np.int0(points)
    area_pred = cv2.contourArea(points)
    break

  area_mask = 0
  contour_mask = find_contours(mask)
  for c in contour_mask:
    if cv2.contourArea(c) < 10:
        continue
    box_mask = cv2.minAreaRect(c)
    points = cv2.boxPoints(box_mask)
    points = np.int0(points)
    area_mask = cv2.contourArea(points)
    break

  if area_mask > 0 and area_pred > 0:
    inter = cv2.rotatedRectangleIntersection(box_pred, box_mask)
    if inter[0] > 0 :
      inter_area=cv2.contourArea(inter[1])
      iou = inter_area / (area_pred + area_mask - inter_area)
      return [inter_area, area_pred, area_mask ,iou]
  return [0,0,0,0]

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


def get_corner(grayimg):
    """角点测试 Demo
        grayimg :为一个灰度图
        返回 res
            角点坐标，1,2是亚像素角点， 3,4是角点
    """
    gray = np.float32(grayimg)
    cornerP = cv2.cornerHarris(gray, 3, 5, 0.04)
    cornerP[cornerP > 0] = 255
 
 
    #cornerP=cv2.dilate(cornerP,None)    #膨胀
    ret, cornerP = cv2.threshold(cornerP, 0.01 * cornerP.max(), 255, 0) #阈值化二值化
    cornerP = np.uint8(cornerP)
    #cv2.imshow('cornerP', cornerP)
    # 图像连通域
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(cornerP) #？？？？？？？？？？
    # 迭代停止规则
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria) #计算亚像素角点
    res = np.hstack((centroids[1:], corners[1:])) #列表 在水平方向上平铺
    res = np.int0(res)   #列表取整
    return np.array(res)

def metric_bounding_rect(mask, gt_y):
    if 0 == np.count_nonzero(mask): 
        return [0,0,0,0,0,0,0,0]

    image = np.uint8(mask * 255)

    img_canny = cv2.Canny(image, 100, 300)
    ret, thresh = cv2.threshold(img_canny, 127, 255, cv2.THRESH_BINARY)
    img_canny = thresh

    # binary是最后返回的二值图像
    #findContours()第一个参数是源图像、第二个参数是轮廓检索模式，第三个参数是轮廓逼近方法
    #输出是轮廓和层次结构，轮廓是图像中所有轮廓的python列表，每个单独的轮廓是对象边界点的(x,y)坐标的Numpy数组
    #binary, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #加粗
    img_canny = cv2.drawContours(img_canny, contours, -1, (255,255,255), 4)
    #binary, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    overlap = np.zeros(mask.shape, dtype = np.uint8)
    #overlay
    overlap = cv2.drawContours(overlap, contours, -1, (255, 255, 255), 1)

    x=0
    y=0
    w=0
    h=0
    center_x = 0
    center_y = 0
    center_left_x = 0
    center_right_x = 0

    blank_line = np.zeros(mask.shape, dtype = np.uint8)

    for c in contours:
        if cv2.contourArea(c) < 10:
                continue
        #x, y, w, h = cv2.boundingRect(c)
        #cv2.rectangle(img_ori, (x,y), (x+w,y+h), (0,255,0), 1)
        x, y ,w, h = cv2.boundingRect(c)
        d = w
        # 计算中心交点
        center_x = int(x + w/2)
        center_y = int(y + h/2)

        if gt_y > 0:
            cv2.line(blank_line, (0,gt_y),(580,gt_y), (255, 255, 255), 1)
        else:
            cv2.line(blank_line, (0,center_y),(580,center_y), (255, 255, 255), 1)

        dst=cv2.bitwise_and(blank_line,overlap)  #轮廓和圆的 与运算
        dst=cv2.dilate(dst, None)               #膨胀获得的点
        res=get_corner(dst)                     #获得点的角点坐标
        if len(res) > 0:
            #图像划线
            center_left_x = np.min(res[:,0])
            center_right_x = np.max(res[:,0])
    
        # find first only
        break
    return [x,y,w,h,center_x,center_y,center_left_x,center_right_x]

def find_contours(img):
  img = np.uint8(img*255.)
  img_canny = cv2.Canny(img, 100, 300)
  ret, thresh = cv2.threshold(img_canny, 127, 255, cv2.THRESH_BINARY)
  img_canny = thresh

  # binary是最后返回的二值图像
  #findContours()第一个参数是源图像、第二个参数是轮廓检索模式，第三个参数是轮廓逼近方法
  #输出是轮廓和层次结构，轮廓是图像中所有轮廓的python列表，每个单独的轮廓是对象边界点的(x,y)坐标的Numpy数组
  binary, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  img_canny = cv2.drawContours(img_canny, contours, -1, (255,255,255), 4)
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
  filenames = glb('../data/score/score_final_test_100/*')
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
    truth_list = [s for s in truth_filename if key+'_mask' in s]
    if len(truth_list)>0:
      pair_filename.append((image, truth_list[0]))
  
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
    csvfile = open('prediction_score_final_test_100_1144490.csv', 'w') 
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['sort','img','truth','tp','tn','fp','fn','tpr','tnr','fpr','fnr','acc','prec','loss','dice','jaccard','f1','f2','bacc','asd','msd','assd','hd','center_x','center_y','rect_w','rect_h','radius','pred_center_x','pred_center_y','pred_rect_w','pred_rect_h','pred_radius','gt_center_x','gt_center_y','gt_rect_w','gt_rect_h','gt_radius','center_dis','and_offset','offset_ratio','bound_inter','bound_pre','bound_mask','bound_iou','min_inter','min_pre','min_mask','min_iou','pred_center_x','pred_center_y','gt_center_x','gt_center_y','pred_left_x','pred_right_x','gt_left_x','gt_right_x','left_edge_dis','right_edge_dis','prj_left_edge_dis','prj_right_edge_dis','hori_left_edge_dis','hori_right_edge_dis'])

    num = 0
    for (f,t) in pair_filename:
      num = num + 1
      # name to save 
      prediction_path = '../data/prediction/'
      name = os.path.split(f)[-1]
      name = os.path.splitext(name)[0]
      name_truth = os.path.split(t)[-1]
      name_truth = os.path.splitext(name)[0]
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
      
      row = [num]
      row.append(name)
      row.append(name_truth)
      row.extend(metric_image_score(generated_mask,ground_truth_mask))
      row.extend(metric_image_distance(generated_mask,ground_truth_mask))
      
      overlay_param = metric_min_area_rect_circle(generated_mask,ground_truth_mask)
      print(overlay_param)
      row.extend(overlay_param)

      overlay_param_pred = metric_img_mar_circle(generated_mask)
      overlay_param_gt = metric_img_mar_circle(ground_truth_mask)
      
      row.extend(overlay_param_pred)
      row.extend(overlay_param_gt)

#      print(metric_bounding_rect_iou(generated_mask,ground_truth_mask))
#      print(metric_min_area_rect_iou(generated_mask,ground_truth_mask))

      x = int(overlay_param[0])
      y = int(overlay_param[1])
      #r = 30 # 20px
      r = int(overlay_param[4])

      x1 = int(overlay_param_pred[0])
      y1 = int(overlay_param_pred[1])
      r1 = int(overlay_param_pred[4])

      x2 = int(overlay_param_gt[0])
      y2 = int(overlay_param_gt[1])
      r2 = int(overlay_param_gt[4])      
     
      # distance
      center_dis = math.sqrt(((x1-x2)**2)+((y1-y2)**2))+0.00001
      and_offset = center_dis - 2 * r
      offset_ratio = and_offset / center_dis

      row.extend([center_dis,and_offset,offset_ratio])

      row.extend(metric_bounding_rect_iou(generated_mask,ground_truth_mask))
      row.extend(metric_min_area_rect_iou(generated_mask,ground_truth_mask))


      bound_truth = metric_bounding_rect(ground_truth_mask,0)
      bound_mask = metric_bounding_rect(generated_mask,0)
      bound_mask_truth = metric_bounding_rect(generated_mask,bound_truth[5])
        
      left_edge_dis = math.sqrt((bound_mask[6] - bound_truth[6])*(bound_mask[6] - bound_truth[6]) \
            + (bound_mask[5] - bound_truth[5])*(bound_mask[5] - bound_truth[5]))
      right_edge_dis = math.sqrt((bound_mask[7] - bound_truth[7])*(bound_mask[7] - bound_truth[7]) \
            + (bound_mask[5] - bound_truth[5])*(bound_mask[5] - bound_truth[5]))

      proj_left_edge_dis = math.fabs(bound_mask[6] - bound_truth[6])
      proj_right_edge_dis = math.fabs(bound_mask[7] - bound_truth[7])

      hori_left_edge_dis = math.fabs(bound_mask_truth[6] - bound_truth[6])
      hori_right_edge_dis =math.fabs(bound_mask_truth[7] - bound_truth[7])

      row.extend([bound_mask[4],bound_mask[5],bound_truth[4],bound_truth[5],bound_mask[6],bound_mask[7], \
                    bound_truth[6],bound_truth[7],left_edge_dis,right_edge_dis, \
                    proj_left_edge_dis,proj_right_edge_dis,hori_left_edge_dis,hori_right_edge_dis])

 
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
    
      
      save_prediction_path = '../data/prediction_score_final_test_100_1144490/'
      if not os.path.exists(save_prediction_path):
         os.mkdir(save_prediction_path)

      filepath_pred = "%s%s.pred.%s"%(save_prediction_path,name,f[-3:])
      filepath_mask = "%s%s.pred.mask.%s"%(save_prediction_path,name,f[-3:])
      #filepath_overlap = "%s%s.overlap.mask.%s"%(save_prediction_path,name,f[-3:])
      filepath_overlap = "%s%s.overlap.mask.%s"%(save_prediction_path,name,'png')
      
      overlap = img_origin.copy()
      contours = find_contours(ground_truth_mask)
      #overlay
      overlap = cv2.drawContours(overlap, contours, -1, (255, 0 , 0), 1)

      contours = find_contours(generated_mask)
      #overlay
      overlap = cv2.drawContours(overlap, contours, -1, (0, 255 , 0), 1)

      #if r > 10 :
      if x > 0 and y > 0:
        cv2.circle(overlap, (x,y), r, (0, 0, 255), 1)
        #cv2.circle(overlap, (x,y), 30, (0, 0, 224), 1) 
        #cv2.line(overlap, (x-r,y),(x+r,y), (0, 0, 255), 1)
        #cv2.line(overlap, (x,y-r),(x,y+r), (0, 0, 255), 1)


      #overlay_param_pred = metric_img_mar_circle(generated_mask)
      #overlay_param_gt = metric_img_mar_circle(ground_truth_mask)
      
      if x1 > 0 and y1 > 0:
        cv2.circle(overlap, (x1,y1), r1, (0, 255, 0), 1)
        #cv2.circle(overlap, (x1,y1), 30, (0, 224, 0), 1)
        #cv2.line(overlap, (x1-r1,y1),(x1+r1,y1), (0, 0, 255), 1)
        #cv2.line(overlap, (x,y-r),(x,y+r), (0, 0, 255), 1)

      if x2 > 0 and y2 > 0:
        cv2.circle(overlap, (x2,y2), r2, (255, 0, 0), 1)
        #cv2.circle(overlap, (x2,y2), 30, (224, 0, 0), 1)
        #cv2.line(overlap, (x2-r2,y2),(x2+r2,y2), (0, 0, 255), 1)
        #cv2.line(overlap, (x2,y2-r2),(x2,y2+r2), (0, 0, 255), 1)
      
      # connect center
      if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
        cv2.line(overlap, (x1,y1),(x2,y2), (0, 0, 255), 1)
      

      # convert to display 
      #generated_mask = np.uint8(generated_mask * 255)

      overlap = cv2.resize(overlap,(580,420),interpolation=cv2.INTER_CUBIC)

      #cv2.imwrite(filepath_pred, img)
      #cv2.imwrite(filepath_mask, generated_mask)
      
      # commen if no need to write
      #cv2.imwrite(filepath_overlap, overlap)
      

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
