

import numpy as np 
import tensorflow as tf 
import cv2 
from glob import glob as glb
import re

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_bool('debug', False,
                            """ this will show the images while generating records. """)

# helper function
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write2TFRecord(image, mask):
  # process frame for saving
  image = np.uint8(image)
  mask = np.uint8(mask)
  image = image.reshape([1,shape[0]*shape[1]])
  mask = mask.reshape([1,shape[0]*shape[1]])
  image = image.tostring()
  mask = mask.tostring()

  # create example and write it
  example = tf.train.Example(features=tf.train.Features(feature={
    'image': _bytes_feature(image),
    'mask': _bytes_feature(mask)})) 
  writer.write(example.SerializeToString()) 

#def write2TFRecord(image, mask):
#  cv2.imshow('image', image) 
#  cv2.waitKey(0)
#  cv2.imshow('mask', mask)
#  cv2.waitKey(0) 

# create tf writer
record_filename = '../data/tfrecords/train.tfrecords'

writer = tf.python_io.TFRecordWriter(record_filename)

cols = 420
rows = 580

# the stored frames
shape = (cols, rows)
frames = np.zeros((shape[0], shape[1], 1))

# list of files
train_filename = glb('../data/train/*') 
mask_filename = [s for s in train_filename if "mask" in s]
image_filename = [s for s in train_filename if "mask" not in s]

pair_filename = []

for image in image_filename:
  key = image[:-4] 
  mask = [s for s in mask_filename if key+'_mask' in s][0] #might some bug happens
  pair_filename.append((image, mask))

for pair in pair_filename:
  # read in images
  image = cv2.imread(pair[0], 0) 
  mask = cv2.imread(pair[1], 0) 
  
  # Display the resulting frame
  if FLAGS.debug == True:
    cv2.imshow('image', image) 
    cv2.waitKey(0)
    cv2.imshow('image', mask) 
    cv2.waitKey(0)
  
  # reinforcement 
  # origin
  write2TFRecord(image,mask)

  # move left
  image_left_20 = np.zeros((cols,rows), np.uint8)
  mask_left_20 = np.zeros((cols,rows), np.uint8)
  image_left_20[20:,:] = image[:-20,:]
  mask_left_20[20:,:] = mask[:-20,:]
  write2TFRecord(image_left_20, mask_left_20)

  # move right
  image_right_20 = np.zeros((cols,rows), np.uint8)
  mask_right_20 = np.zeros((cols,rows), np.uint8)
  image_right_20[0:-20,:] = image[20:,:]
  mask_right_20[0:-20,:] = mask[20:,:]
  write2TFRecord(image_right_20, mask_right_20)

  # move left
  image_left_40 = np.zeros((cols,rows), np.uint8)
  mask_left_40 = np.zeros((cols,rows), np.uint8)
  image_left_40[40:,:] = image[:-40,:]
  mask_left_40[40:,:] = mask[:-40,:]
  write2TFRecord(image_left_40, mask_left_40)

  # move right
  image_right_40 = np.zeros((cols,rows), np.uint8)
  mask_right_40 = np.zeros((cols,rows), np.uint8)
  image_right_40[0:-40,:] = image[40:,:]
  mask_right_40[0:-40,:] = mask[40:,:]
  write2TFRecord(image_right_40, mask_right_40)

  # move left 
  image_left_60 = np.zeros((cols,rows), np.uint8)
  mask_left_60 = np.zeros((cols,rows), np.uint8)
  image_left_60[60:,:] = image[:-60,:]
  mask_left_60[60:,:] = mask[:-60,:]
  write2TFRecord(image_left_60, mask_left_60)

  # move right
  image_right_60 = np.zeros((cols,rows), np.uint8)
  mask_right_60 = np.zeros((cols,rows), np.uint8)
  image_right_60[0:-60,:] = image[60:,:]
  mask_right_60[0:-60,:] = mask[60:,:]
  write2TFRecord(image_right_60, mask_right_60)

  # move left top 
  image_left_top_20 = np.zeros((cols,rows), np.uint8)
  mask_left_top_20 = np.zeros((cols,rows), np.uint8)
  image_left_top_20[20:,20:] = image[:-20,:-20]
  mask_left_top_20[20:,20:] = mask[:-20,:-20]
  write2TFRecord(image_left_top_20, mask_left_top_20)

  # move right bottom
  image_right_bottom_20 = np.zeros((cols,rows), np.uint8)
  mask_right_bottom_20 = np.zeros((cols,rows), np.uint8)
  image_right_bottom_20[0:-20,0:-20] = image[20:,20:]
  mask_right_bottom_20[0:-20,0:-20] = mask[20:,20:]
  write2TFRecord(image_right_bottom_20, mask_right_bottom_20)

  # move left bottom
  image_left_bottom_20 = np.zeros((cols,rows), np.uint8)
  mask_left_bottom_20 = np.zeros((cols,rows), np.uint8)
  image_left_bottom_20[0:-20,20:] = image[20:,0:-20]
  mask_left_bottom_20[0:-20,20:] = mask[20:,0:-20]
  write2TFRecord(image_left_bottom_20, mask_left_bottom_20)

  # move right top
  image_right_top_20 = np.zeros((cols,rows), np.uint8)
  mask_right_top_20 = np.zeros((cols,rows), np.uint8)
  image_right_top_20[20:,0:-20] = image[0:-20,20:]
  mask_right_top_20[20:,0:-20] = mask[0:-20,20:]
  write2TFRecord(image_right_top_20, mask_right_top_20)

  # move left top 
  image_left_top_40 = np.zeros((cols,rows), np.uint8)
  mask_left_top_40 = np.zeros((cols,rows), np.uint8)
  image_left_top_40[40:,40:] = image[:-40,:-40]
  mask_left_top_40[40:,40:] = mask[:-40,:-40]
  write2TFRecord(image_left_top_40, mask_left_top_40)

  # move right bottom
  image_right_bottom_40 = np.zeros((cols,rows), np.uint8)
  mask_right_bottom_40 = np.zeros((cols,rows), np.uint8)
  image_right_bottom_40[0:-40,0:-40] = image[40:,40:]
  mask_right_bottom_40[0:-40,0:-40] = mask[40:,40:]
  write2TFRecord(image_right_bottom_40, mask_right_bottom_40)

  # move left bottom
  image_left_bottom_40 = np.zeros((cols,rows), np.uint8)
  mask_left_bottom_40 = np.zeros((cols,rows), np.uint8)
  image_left_bottom_40[0:-40,40:] = image[40:,0:-40]
  mask_left_bottom_40[0:-40,40:] = mask[40:,0:-40]
  write2TFRecord(image_left_bottom_40, mask_left_bottom_40)

  # move right top
  image_right_top_40 = np.zeros((cols,rows), np.uint8)
  mask_right_top_40 = np.zeros((cols,rows), np.uint8)
  image_right_top_40[40:,0:-40] = image[0:-40,40:]
  mask_right_top_40[40:,0:-40] = mask[0:-40,40:]
  write2TFRecord(image_right_top_40, mask_right_top_40)

  # flip vertical
  image_flip_v = cv2.flip(image, 1)
  mask_flip_v = cv2.flip(mask, 1)
  write2TFRecord(image_flip_v, mask_flip_v)

  # flip horizional
  image_flip_h = cv2.flip(image, 2)
  mask_flip_h = cv2.flip(mask, 2)
  write2TFRecord(image_flip_h, mask_flip_h)

  # 按照比例缩放，如x,y轴均放大
  image_resize = cv2.resize(image, (rows+40,cols+40))
  mask_resize = cv2.resize(mask, (rows+40,cols+40))
  image_corp = image_resize[20:cols+20,20:rows+20]
  mask_corp = mask_resize[20:cols+20,20:rows+20]
  write2TFRecord(image_corp, mask_corp)

  # 按照比例缩放，如x,y轴均放大
  image_resize = cv2.resize(image, (rows-40,cols-40))
  mask_resize = cv2.resize(mask, (rows-40,cols-40))
  image_padding = np.zeros((cols,rows), np.uint8)
  mask_padding = np.zeros((cols,rows), np.uint8)
  image_padding[20:cols-20,20:rows-20] = image_resize
  mask_padding[20:cols-20,20:rows-20] = mask_resize
  write2TFRecord(image_padding, mask_padding)

  # Rotation Matrix2D
  matrix_5 = cv2.getRotationMatrix2D((rows / 2 , cols / 2), 5, 1.1)
  image_matrix_5 = cv2.warpAffine(image, matrix_5, (rows, cols))
  mask_matrix_5 = cv2.warpAffine(mask, matrix_5, (rows, cols))
  write2TFRecord(image_matrix_5, mask_matrix_5)

  # Rotation Matrix2D
  matrix_m5 = cv2.getRotationMatrix2D((rows / 2 , cols / 2), -5, 1.1)
  image_matrix_m5 = cv2.warpAffine(image, matrix_m5, (rows, cols))
  mask_matrix_m5 = cv2.warpAffine(mask, matrix_m5, (rows, cols))
  write2TFRecord(image_matrix_m5, mask_matrix_m5)

  # Rotation Matrix2D
  matrix_10 = cv2.getRotationMatrix2D((rows / 2 , cols / 2), 10, 1.1)
  image_matrix_10 = cv2.warpAffine(image, matrix_10, (rows, cols))
  mask_matrix_10 = cv2.warpAffine(mask, matrix_10, (rows, cols))
  write2TFRecord(image_matrix_10, mask_matrix_10)

  # Rotation Matrix2D
  matrix_m10 = cv2.getRotationMatrix2D((rows / 2 , cols / 2), -10, 1.1)
  image_matrix_m10 = cv2.warpAffine(image, matrix_m10, (rows, cols))
  mask_matrix_m10 = cv2.warpAffine(mask, matrix_m10, (rows, cols))
  write2TFRecord(image_matrix_m10, mask_matrix_m10)
