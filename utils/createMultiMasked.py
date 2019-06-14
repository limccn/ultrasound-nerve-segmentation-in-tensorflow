
import cv2 
import numpy as np 
from glob import glob as glb
import re


# get a list of image filenames
filenames = glb('../data/step-valid/test/*')
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

    print(pair[0])
    print(pair[1])
    
    mask1_path=str(pair[0]).replace('test','mask-50000').replace('.tif','.pred.mask.tif')
    mask1=cv2.imread(mask1_path, cv2.IMREAD_COLOR)
    mask1_gray=cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)

    mask2_path=str(pair[0]).replace('test','mask-100000').replace('.tif','.pred.mask.tif')
    mask2=cv2.imread(mask2_path, cv2.IMREAD_COLOR)
    mask2_gray=cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

    mask3_path=str(pair[0]).replace('test','mask-250000').replace('.tif','.pred.mask.tif')
    mask3=cv2.imread(mask3_path, cv2.IMREAD_COLOR)
    mask3_gray=cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)

    mask4_path=str(pair[0]).replace('test','mask-400000').replace('.tif','.pred.mask.tif')
    mask4=cv2.imread(mask4_path, cv2.IMREAD_COLOR)
    mask4_gray=cv2.cvtColor(mask4, cv2.COLOR_BGR2GRAY)


    # 3*3 GaussianBlur
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # canny detect edge
    gray = cv2.Canny(gray, 100, 300)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # binary是最后返回的二值图像
    #findContours()第一个参数是源图像、第二个参数是轮廓检索模式，第三个参数是轮廓逼近方法
    #输出是轮廓和层次结构，轮廓是图像中所有轮廓的python列表，每个单独的轮廓是对象边界点的(x,y)坐标的Numpy数组
    binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 255, 255), 2)

    # 3*3 GaussianBlur
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # canny detect edge
    mask1_gray = cv2.Canny(mask1_gray, 100, 300)

    ret, thresh = cv2.threshold(mask1_gray, 127, 255, cv2.THRESH_BINARY)

    # binary是最后返回的二值图像
    #findContours()第一个参数是源图像、第二个参数是轮廓检索模式，第三个参数是轮廓逼近方法
    #输出是轮廓和层次结构，轮廓是图像中所有轮廓的python列表，每个单独的轮廓是对象边界点的(x,y)坐标的Numpy数组
    binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)


    # 3*3 GaussianBlur
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # canny detect edge
    mask2_gray = cv2.Canny(mask2_gray, 100, 300)

    ret, thresh = cv2.threshold(mask2_gray, 127, 255, cv2.THRESH_BINARY)

    # binary是最后返回的二值图像
    #findContours()第一个参数是源图像、第二个参数是轮廓检索模式，第三个参数是轮廓逼近方法
    #输出是轮廓和层次结构，轮廓是图像中所有轮廓的python列表，每个单独的轮廓是对象边界点的(x,y)坐标的Numpy数组
    binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 0, 0), 1)

    # 3*3 GaussianBlur
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # canny detect edge
    mask3_gray = cv2.Canny(mask3_gray, 100, 300)

    ret, thresh = cv2.threshold(mask3_gray, 127, 255, cv2.THRESH_BINARY)

    # binary是最后返回的二值图像
    #findContours()第一个参数是源图像、第二个参数是轮廓检索模式，第三个参数是轮廓逼近方法
    #输出是轮廓和层次结构，轮廓是图像中所有轮廓的python列表，每个单独的轮廓是对象边界点的(x,y)坐标的Numpy数组
    binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1)

    
    # 3*3 GaussianBlur
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # canny detect edge
    mask4_gray = cv2.Canny(mask4_gray, 100, 300)

    ret, thresh = cv2.threshold(mask4_gray, 127, 255, cv2.THRESH_BINARY)

    # binary是最后返回的二值图像
    #findContours()第一个参数是源图像、第二个参数是轮廓检索模式，第三个参数是轮廓逼近方法
    #输出是轮廓和层次结构，轮廓是图像中所有轮廓的python列表，每个单独的轮廓是对象边界点的(x,y)坐标的Numpy数组
    binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 255, 0), 1)
    

    #
    # cv2.imwrite('%s.1.png'%pair[1],binary,[int(cv2.IMWRITE_PNG_COMPRESSION),3])
    cv2.imwrite('%s.masked.png'%str(pair[0]).replace('test','output'),image,[int(cv2.IMWRITE_PNG_COMPRESSION),3])
    