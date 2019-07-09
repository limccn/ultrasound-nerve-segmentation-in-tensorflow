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

import os

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import model.nerve_net as nerve_net 
import input.nerve_input as nerve_input
from run_length_encoding import RLenc
from utils.experiment_manager import make_checkpoint_path

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', '../checkpoints',
                            """dir to store trained net """)
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('max_steps', 510000,
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


def image_preprocess(image):
  cols = 420
  rows = 580
  corp_cols = 210
  corp_rows = 290
  offset_cols=80
  offset_rows=150

  image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_corp = np.zeros((corp_cols,corp_rows), np.uint8)
  #corp
  image_corp = image[80:80+corp_cols,150:150+corp_rows]
  #resize
  image = cv2.resize(image_corp, (rows,cols))

  return image

def image_reinforce(image):
  img_inforce=dict()
  # brighness + contrast
  # g(x) = br * f(x) + co
  for br in range(5, 15):
    for co in range(0, 10) :
      res = np.uint8(np.clip((br/10. * image + co*10), 0, 255))
      key = "br"+str(br)+"_co"+str(co)
      img_inforce[key] = res
  
def video_split():
  path = '../data/video_test/*'
  
  video_list = glb(path) #获取该目录下的所有文件名
  video_list.sort(key=alphanum_key)
  print(video_list)

  for video_path in video_list:
    print(video_path)
    video_name = video_path[:-4]
    
    #mkdir
    if not os.path.isdir(video_name):
        os.makedirs(video_name)

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    print(fps)
    count = 0
    while success:
         #mkdir
        #subdir = "%s/s%df%d" % (video_name,int(count/fps),count%fps)
        #if not os.path.isdir(subdir):
        #    os.makedirs(subdir)
        #preprocess
        subdir= video_name
        image = image_preprocess(image)
        cv2.imwrite("%s/%d.tif" % (subdir, count), image)
        #if count % fps == 0:
        #    cv2.imwrite("%s/%d.jpg" % (video_name, int(count / fps)), image)
        #print('Process %dth seconds: ' % int(count / fps), success)
        success,image = vidcap.read()
        count += 1


# 图片合成视频
def video_combine():
    path = '../data/test/video_test/video_01/*'#文件路径
    filelist = glb(path) #获取该目录下的所有文件名
    filelist.sort(key=alphanum_key)

    '''
    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
    '''
    fps = 24
    size = (580,420) #图片的分辨率片
    file_path = 'test/video/video_01_out.mp4'#导出路径
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')#不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    video = cv2.VideoWriter( file_path, fourcc, fps, size)

    for item in filelist:
        if item.endswith('.jpg'):   #判断图片后缀是否是.png
            item = path + '/' + item 
            image = cv2.imread(item)  #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            video.write(image)        #把图片写进视频
    
    video.release() #释放


def evaluate():
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  # get a list of image filenames
  filenames = glb('../data/rein_test/*')
  # sort the file names but this is probably not ness
  filenames.sort(key=alphanum_key)
  #num_files = len(filename)

  with tf.Graph().as_default():
    # Make image placeholder
    images_op = tf.placeholder(tf.float32, [1, 420, 580, 1])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    mask = nerve_net.inference(images_op,1.0)

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    
    sess = tf.Session()

    ckpt = tf.train.get_checkpoint_state(TEST_DIR)

    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = 1
    
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    #summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
    #                                        graph_def=graph_def)

    # make csv file
    #csvfile = open('test.csv', 'wb') 
    csvfile = open('test.csv', 'w') 
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['img', 'pixels'])

    for f in filenames:
      # name to save 
      prediction_path = '../data/prediction/'
      name = f[13:-4]
      print(name)
     
      # read in image
      img = cv2.imread(f, 0)
      img_inforce=dict()
      # brighness + contrast
      # g(x) = br * f(x) + co
      for co in range(6, 20):
        con = co/10.
        bri_base = int(125*(1-con))
        for br in range(2, 22) :
          bri = bri_base + (br * 10) - 50
          res = np.uint8(np.clip((con * img + bri), 0, 255))
          key = "br_"+str(br)+"co_"+str(co)
          print(key)
          img_inforce[key] = res

      for (key,img_inf) in img_inforce.items() :
        print(key)
        img_inf = img_inf - np.mean(img_inf)
  
        # format image for network
        img_inf = np.expand_dims(img_inf, axis=0)
        img_inf = np.expand_dims(img_inf, axis=3)
    
        # calc logits 
        generated_mask = sess.run([mask],feed_dict={images_op: img_inf})
        generated_mask = generated_mask[0]
        generated_mask = generated_mask[0, :, :, :]
     
     
        # bin for converting to row format
        threshold = .5
        generated_mask[:][generated_mask[:]<=threshold]=0 
        generated_mask[:][generated_mask[:]>threshold]=1 
        run_length_encoding = RLenc(generated_mask)
        print(run_length_encoding)
        #name = name.encode('utf8')
        #run_length_encoding = run_length_encoding.encode('utf8')
        #writer.writerow([name, run_length_encoding])
     
        save_prediction_path = '../data/rein_pred_save/'
        filepath_pred = "%s%s_%s_pred.%s"%(save_prediction_path,name,key,f[-3:])
        filepath_mask = "%s%s_%s_mask.%s"%(save_prediction_path,name,key,f[-3:])
      
        print(filepath_pred)
        print(filepath_mask)

        # convert to display 
        generated_mask = np.uint8(generated_mask * 255)
        cv2.imwrite(filepath_pred, img_inf[0,:,:,0])
        cv2.imwrite(filepath_mask, generated_mask[:,:,0])
        
        generated_mask = np.uint8(generated_mask)

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
    


      if False: 
        # display image
        cv2.imshow('img', np.uint8(img[0,:,:,0]*255.0))
        cv2.waitKey(0)
        cv2.imshow('mask', generated_mask[:,:,0]*255)
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break  
    

def main(argv=None):  # pylint: disable=unused-argument
  #evaluate()
  video_split()

if __name__ == '__main__':
  tf.app.run()

