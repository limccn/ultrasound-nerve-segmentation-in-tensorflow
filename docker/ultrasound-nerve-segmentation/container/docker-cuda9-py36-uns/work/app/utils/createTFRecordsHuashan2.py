

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
  
  # change image size
  image = cv2.resize(image, (290,210),interpolation=cv2.INTER_CUBIC)
  mask = cv2.resize(mask, (290,210),interpolation=cv2.INTER_CUBIC)
  
  # process frame for saving
  image = np.uint8(image)
  mask = np.uint8(mask)
  image = image.reshape([1,shape[0]//2*shape[1]//2])
  mask = mask.reshape([1,shape[0]//2*shape[1]//2])
  image = image.tostring()
  mask = mask.tostring()

  # create example and write it
  example = tf.train.Example(features=tf.train.Features(feature={
    'image': _bytes_feature(image),
    'mask': _bytes_feature(mask)})) 
  writer.write(example.SerializeToString())
  global record_count
  record_count = record_count+1

#def write2TFRecord(image, mask):
#  cv2.imshow('image', image) 
#  cv2.waitKey(0)
#  cv2.imshow('mask', mask)
#  cv2.waitKey(0) 

# create tf writer
record_filename = '../data/tfrecords/train_huashan.tfrecords'
record_count=0

writer = tf.python_io.TFRecordWriter(record_filename)

cols = 420
rows = 580

# the stored frames
shape = (cols, rows)
frames = np.zeros((shape[0], shape[1], 1))

# list of files
train_filename = glb('../data/train/train_huashan/*') 
mask_filename = [s for s in train_filename if "mask" in s]
image_filename = [s for s in train_filename if "mask" not in s]

pair_filename = []

for image in image_filename:
  key = image[:-4] 
  mask = [s for s in mask_filename if key+'_mask' in s][0] #might some bug happens
  pair_filename.append((image, mask))

print ("file_count:%d"%len(pair_filename))

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

  
  for px in range(10,40,10):
    # move top
    image_top = np.zeros((cols,rows), np.uint8)
    mask_top = np.zeros((cols,rows), np.uint8)
    image_top[px:,:] = image[:-px,:]
    mask_top[px:,:] = mask[:-px,:]
    write2TFRecord(image_top, mask_top)

  for px in range(10,60,10):
    # move bottom
    image_bottom = np.zeros((cols,rows), np.uint8)
    mask_bottom = np.zeros((cols,rows), np.uint8)
    image_bottom[0:-px,:] = image[px:,:]
    mask_bottom[0:-px,:] = mask[px:,:]
    write2TFRecord(image_bottom, mask_bottom)
  
  # move left
  for px in range(10,60,10):
    image_left = np.zeros((cols,rows), np.uint8)
    mask_left = np.zeros((cols,rows), np.uint8)
    image_left[:,px:] = image[:,:-px]
    mask_left[:,px:] = mask[:,:-px]
    write2TFRecord(image_left, mask_left)

  # move right
  for px in range(10,60,10):
    image_right = np.zeros((cols,rows), np.uint8)
    mask_right = np.zeros((cols,rows), np.uint8)
    image_right[:,0:-px] = image[:,px:]
    mask_right[:,0:-px] = mask[:,px:]
    write2TFRecord(image_right, mask_right)

  # move left top 
  for px in range(10,40,10):
    image_left_top = np.zeros((cols,rows), np.uint8)
    mask_left_top = np.zeros((cols,rows), np.uint8)
    image_left_top[px:,px:] = image[:-px,:-px]
    mask_left_top[px:,px:] = mask[:-px,:-px]
    write2TFRecord(image_left_top, mask_left_top)

  # move right bottom
  for px in range(10,60,10):
    image_right_bottom = np.zeros((cols,rows), np.uint8)
    mask_right_bottom = np.zeros((cols,rows), np.uint8)
    image_right_bottom[0:-px,0:-px] = image[px:,px:]
    mask_right_bottom[0:-px,0:-px] = mask[px:,px:]
    write2TFRecord(image_right_bottom, mask_right_bottom)

  # move left bottom
  for px in range(10,60,10):
    image_left_bottom = np.zeros((cols,rows), np.uint8)
    mask_left_bottom = np.zeros((cols,rows), np.uint8)
    image_left_bottom[0:-px,px:] = image[px:,0:-px]
    mask_left_bottom[0:-px,px:] = mask[px:,0:-px]
    write2TFRecord(image_left_bottom, mask_left_bottom)

  # move right top
  for px in range(10,40,10):
    image_right_top = np.zeros((cols,rows), np.uint8)
    mask_right_top = np.zeros((cols,rows), np.uint8)
    image_right_top[px:,0:-px] = image[0:-px,px:]
    mask_right_top[px:,0:-px] = mask[0:-px,px:]
    write2TFRecord(image_right_top, mask_right_top)
  
  # 按照比例缩放，如x,y轴均放大
  for px in range(10,40,10):
    image_resize = cv2.resize(image, (rows+px*2,cols+px*2))
    mask_resize = cv2.resize(mask, (rows+px*2,cols+px*2))
    image_corp = image_resize[px:cols+px,px:rows+px]
    mask_corp = mask_resize[px:cols+px,px:rows+px]
    write2TFRecord(image_corp, mask_corp)

  # 按照比例缩放，如x,y轴均放大
  for px in range(10,40,10):
    image_resize = cv2.resize(image, (rows-px*2,cols-px*2))
    mask_resize = cv2.resize(mask, (rows-px*2,cols-px*2))
    image_padding = np.zeros((cols,rows), np.uint8)
    mask_padding = np.zeros((cols,rows), np.uint8)
    image_padding[px:cols-px,px:rows-px] = image_resize
    mask_padding[px:cols-px,px:rows-px] = mask_resize
    write2TFRecord(image_padding, mask_padding)

  # Rotation Matrix2D
  for px in range(5,25,4):
    matrix = cv2.getRotationMatrix2D((rows / 2 , cols / 2), px, 1.1)
    image_matrix = cv2.warpAffine(image, matrix, (rows, cols))
    mask_matrix = cv2.warpAffine(mask, matrix, (rows, cols))
    write2TFRecord(image_matrix, mask_matrix)

  # Rotation Matrix2D
  for px in range(5,25,4):
    matrix_m = cv2.getRotationMatrix2D((rows / 2 , cols / 2), -px, 1.1)
    image_matrix_m = cv2.warpAffine(image, matrix_m, (rows, cols))
    mask_matrix_m = cv2.warpAffine(mask, matrix_m, (rows, cols))
    write2TFRecord(image_matrix_m, mask_matrix_m)

  # flip vertical
  image_flip_v = cv2.flip(image, 1)
  mask_flip_v = cv2.flip(mask, 1)
  write2TFRecord(image_flip_v, mask_flip_v)

  # flip horizional
  image_flip_h = cv2.flip(image, 2)
  mask_flip_h = cv2.flip(mask, 2)
  write2TFRecord(image_flip_h, mask_flip_h)

print ("record_count:%d"%record_count)

