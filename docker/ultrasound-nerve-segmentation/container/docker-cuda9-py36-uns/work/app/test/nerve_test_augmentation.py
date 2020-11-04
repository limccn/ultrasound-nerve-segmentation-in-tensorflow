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

from PIL import ImageDraw
from PIL import ImageChops
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import Image

import skimage

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
tf.app.flags.DEFINE_integer('max_steps', 227980,
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

def cv2_to_pil(image):
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))  
    return image

def pil_to_cv2(image):
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
    return image

def ski_to_pil(image):
    return cv2_to_pil(ski_to_cv2(image))

def pil_to_ski(image):
    return cv2_to_ski(pil_to_cv2(image))

def ski_to_cv2(image):
    image *= 255.
    image = image.astype('uint8')
    cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image

def cv2_to_ski(image):
    cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.astype(float)
    image /= 255.
    return image

# 图像属性调整
def adjust_image(image, brightness=1.,contrast=1.,sharpness=1.):
    result_image = image.copy()
    (w,h) = result_image.size
    
#   print(brightness,contrast,sharpness)
    # 亮度
    if (brightness - 1) != 0 :
        result_image = ImageEnhance.Brightness(result_image).enhance(brightness)
    # 对比度
    if (contrast - 1) != 0 : 
        result_image = ImageEnhance.Contrast(result_image).enhance(contrast)
    # 锐度
    if (sharpness - 1) != 0 : 
        result_image = ImageEnhance.Sharpness(result_image).enhance(sharpness)
    return result_image

# 曝光度调整
def exposure_image(image,hist_nbins=10, sigmoid=False, gamma_adj=1.):
    result_image = image.copy()
    (w,h) = result_image.size
    result_image = pil_to_ski(result_image)
    # 直方图自适应
    if hist_nbins > 0 :
        result_image = skimage.exposure.equalize_adapthist(result_image,20)
    # sigmod 曲线调整
    if sigmoid :
        result_image = skimage.exposure.adjust_sigmoid(result_image)
    # gamma 调整
    if (gamma_adj - 1) !=0 :
        result_image = skimage.exposure.adjust_gamma(result_image, gamma=gamma_adj, gain=1)
    return ski_to_pil(result_image)

def image_augmentation_adjust_strategy():
    steps = []
    steps.append((1.,1.,1.))
    # 循环产生增强图片
    for brightness in range(5,50,8):
        for contrast in range(5,50,8):
           for sharpness in range(5,50,8): 
               steps.append(((100+brightness)/100.,(100+contrast)/100.,(100+sharpness)/100.))
               steps.append(((100-brightness)/100.,(100-contrast)/100.,(100-sharpness)/100.))
    return steps      

def image_augmentation_exposure_strategy():
    steps = []
    # 循环产生增强图片
    steps.append((False,1.,1.))
    for nbins in range(0,11,1):
       for gamma in range(5,30,5):
           steps.append((False,(20+nbins)/100.,(100+gamma)/100.))
           steps.append((False,(20-nbins)/100.,(100-gamma)/100.))
    steps.append((True,1.,1.))
    for nbins in range(0,11,1):
       for gamma in range(5,30,5):
           steps.append((True,(20+nbins)/100.,(100+gamma)/100.))
           steps.append((True,(20-nbins)/100.,(100-gamma)/100.))
    return steps      

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
  filenames = glb('../data/augmentation/augmentation_batch_4_image/*')
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
    truth_files = [s for s in truth_filename if key+'_mask' in s]
    if len(truth_files) > 0:
        truth = truth_files[0]
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
    csvfile = open('augmentation_huashan_batch_4_image_227980.csv', 'w') 
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['img','truth','brightness','contrast','sharpness','tp','tn','fp','fn','tpr','tnr','fpr','fnr','acc','prec','loss','dice','jaccard','f1','f2','bacc','asd','msd','assd','hd','center_x','center_y','rect_w','rect_h','radius','pred_center_x','pred_center_y','pred_rect_w','pred_rect_h','pred_radius','gt_center_x','gt_center_y','gt_rect_w','gt_rect_h','gt_radius','center_dis','and_offset','offset_ratio'])
    
    adjust_stg = image_augmentation_adjust_strategy()
    print(len(adjust_stg))
    #exposure_stg = image_augmentation_exposure_strategy()
    #print(adjust_stg)
    
    for (f,t) in pair_filename:
      # name to save 
      augmentation_save_path = '../data/augmentation/save/batch_4_image'
      name = os.path.split(f)[-1]
      name = os.path.splitext(name)[0]
      name_truth = os.path.split(t)[-1]
      name_truth = os.path.splitext(name)[0]
      print(name)
     
      # read in image
      img_origin = cv2.imread(f,cv2.IMREAD_COLOR)
      # ground truth
      img_gt = cv2.imread(t, 0) 
      
      best_redius = 0
      best_offset = 999
      best_row = None
      best_param = None
      for adjust_param in adjust_stg:
          (brightness,contrast,sharpness) = adjust_param
          img = img_origin.copy()
          # adjust
          img = cv2_to_pil(img)
          img = adjust_image(img,brightness,contrast,sharpness)
          img = pil_to_cv2(img)

          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          # resize
          img = cv2.resize(img,(290,210),interpolation=cv2.INTER_CUBIC)
      
          img = img - np.mean(img)
 
          # format image for network
          img = np.expand_dims(img, axis=0)
          img = np.expand_dims(img, axis=3)
          
          # ground truth
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
      
          #计算内切半径
          overlay_param = metric_min_area_rect_circle(generated_mask,ground_truth_mask)
          #print(overlay_param)
          
          #计算中心偏移率
          overlay_param_pred = metric_img_mar_circle(generated_mask)
          overlay_param_gt = metric_img_mar_circle(ground_truth_mask)
          #中心偏移
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

          # find best offset, allow 2% 
          if offset_ratio - best_offset <= 0.02:
              best_offset = offset_ratio
              # find best redius allow 4px 
              if overlay_param[4] - best_redius >= -4:
                  #set to best
                  best_redius = overlay_param[4]
                  best_param = adjust_param
                  #make row
                  row=[]
                  row.append(name)
                  row.append(name_truth)
                  row.extend([brightness,contrast,sharpness])
          
                  image_score = metric_image_score(generated_mask,ground_truth_mask)
                  row.extend(image_score)
                  image_distance = metric_image_distance(generated_mask,ground_truth_mask)
                  row.extend(image_distance)
      
                  #overlay_param = metric_min_area_rect_circle(generated_mask,ground_truth_mask)
                  #print(overlay_param)
                  row.extend(overlay_param)

                  #overlay_param_pred = metric_img_mar_circle(generated_mask)
                  row.extend(overlay_param_pred)

                  #overlay_param_gt = metric_img_mar_circle(ground_truth_mask)
                  row.extend(overlay_param_gt)

                  #print([center_dis,and_offset,offset_ratio])
                  row.extend([center_dis,and_offset,offset_ratio])
                  # replace
                  best_row = row
                  #break
              pass
          pass
      
      print((best_redius,best_offset,best_param))    
      writer.writerow(best_row)
      
      img_save = img_origin.copy()
      (brightness,contrast,sharpness) = best_param
      # adjust
      img_save = cv2_to_pil(img_save)
      img_save = adjust_image(img_save,brightness,contrast,sharpness)
      
      save_file_path = os.path.join(augmentation_save_path,"%s.aug.tif"%name)
      print("save to %s"%save_file_path)
      #save
      img_save.save(save_file_path)

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
    
      '''
      save_prediction_path = '../data/prediction_save_new_sample_227980/'
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
      #cv2.imwrite(filepath_mask, generated_mask[:,:,0])
      cv2.imwrite(filepath_overlap, overlap)
      

      generated_mask = np.uint8(generated_mask)
      '''
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
