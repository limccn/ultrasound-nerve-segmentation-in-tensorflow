
import cv2 
import numpy as np 
from glob import glob as glb
import re


# get a list of image filenames
filenames = glb('../data/result/inferent/*')
#num_files = len(filename)

mask_filename = [s for s in filenames if "mask" in s]
image_filename = [s for s in filenames if "mask" not in s]

pair_filename = []

for image in image_filename:
    key = image[:-4] 
    mask = [s for s in mask_filename if key+'_mask' in s][0]
    pair_filename.append((image, mask))

print(pair_filename)

for pair in pair_filename:
    # read in images
    image = cv2.imread(pair[0], cv2.IMREAD_COLOR) 
    mask = cv2.imread(pair[1], cv2.IMREAD_COLOR)
    # convert colorspace
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 3*3 GaussianBlur
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # canny detect edge
    gray = cv2.Canny(gray, 100, 300)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # binary是最后返回的二值图像
    #findContours()第一个参数是源图像、第二个参数是轮廓检索模式，第三个参数是轮廓逼近方法
    #输出是轮廓和层次结构，轮廓是图像中所有轮廓的python列表，每个单独的轮廓是对象边界点的(x,y)坐标的Numpy数组
    binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    #
    # cv2.imwrite('%s.1.png'%pair[1],binary,[int(cv2.IMWRITE_PNG_COMPRESSION),3])
    cv2.imwrite('%s.masked.png'%pair[0],image,[int(cv2.IMWRITE_PNG_COMPRESSION),3])
    