
import cv2 
import numpy as np 
from glob import glob as glb
import re
from matplotlib import pyplot as plt


def motion_blur(image, degree=12, angle=45):
    image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

# get a list of image filenames
filenames = glb('data/rein_test_4/ORIGIN/*')
#num_files = len(filename)

mask_filename = [s for s in filenames if "_mask" in s]
image_filename = [s for s in filenames if "_mask" not in s]

pair_filename = []

for image in image_filename:
    key = image[:-4] 
    mask_files = [s for s in mask_filename if key+'_' in s]
    pair_filename.append((image, mask_files))

print(pair_filename)

for pair in pair_filename:
    # read in images
    image = cv2.imread(pair[0], cv2.IMREAD_COLOR) 
    heatmap = np.zeros(shape=(image.shape[0],image.shape[1]))
    for mask_file in pair[1]:
        mask = cv2.imread(mask_file, cv2.IMREAD_COLOR)
        # convert colorspace
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # threshold
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # 
        heatmap = heatmap + (thresh/255.)

    heatmap_depth= np.max(heatmap) - np.min(heatmap) + 1
    heatmap = (heatmap / heatmap_depth * 255.)
    heatmap = motion_blur(heatmap,degree=20, angle=45)

    heatmap_g = heatmap.astype(np.uint8)
    cv2.imwrite('%s.gray.png'%str(pair[0]),heatmap_g,[int(cv2.IMWRITE_PNG_COMPRESSION),3])

    heatmap_color = cv2.applyColorMap(heatmap_g, cv2.COLORMAP_JET)
    cv2.imwrite('%s.color.png'%str(pair[0]),heatmap_color,[int(cv2.IMWRITE_PNG_COMPRESSION),3])

    '''

    heatmap_r = heatmap.astype(np.float32)

    # red
    heatmap_r[:][heatmap_r[:] < 191]=0.
    heatmap_r[:][heatmap_r[:] > 255]=255.
    heatmap_r = (heatmap_r-127)*2-1
    heatmap_r[:][heatmap_r[:] < 0]=0.
    heatmap_r[:][heatmap_r[:] > 255]=255.
    heatmap_r = np.uint8(heatmap_r)

    # green
    heatmap_g = heatmap.astype(np.float32)
    heatmap_g[:][heatmap_g[:] < 63]=0.
    heatmap_g[:][heatmap_g[:] > 191]=0.
    heatmap_g = (heatmap_g-63)*2-1
    heatmap_g[:][heatmap_g[:] < 0]=0.
    heatmap_r[:][heatmap_r[:] > 255]=255.
    heatmap_g = np.uint8(heatmap_g)

    # blue
    heatmap_b = heatmap.astype(np.float32)
    heatmap_b[:][heatmap_b[:] > 63]=0.
    heatmap_b = heatmap_b*4-1
    heatmap_b[:][heatmap_b[:] < 0]=0.
    heatmap_b[:][heatmap_b[:] > 255]=255.
    heatmap_b = np.uint8(heatmap_b)

    # alpha
    heatmap_a = heatmap.astype(np.float32)+64
    heatmap_a[:][heatmap_a[:] > 255]=255.
    heatmap_a = np.uint8(heatmap_a)

    # heatmap
    heatmap_bgr = cv2.merge([heatmap_b,heatmap_g,heatmap_r])
    cv2.imwrite('%s.heatmap.bgr.png'%str(pair[0]),heatmap_bgr,[int(cv2.IMWRITE_PNG_COMPRESSION),3])
    heatmap_bgra = cv2.merge([heatmap_b,heatmap_g,heatmap_r,heatmap_a])
    cv2.imwrite('%s.heatmap.bgra.png'%str(pair[0]),heatmap_bgra,[int(cv2.IMWRITE_PNG_COMPRESSION),3])
    '''

    # overlay
    merge_img = image.copy()
    heatmap_img = heatmap_color.copy()
    overlay = image.copy()
    alpha = 0.25 # 设置覆盖图片的透明度
    #cv2.rectangle(overlay, (0, 0), (merge_img.shape[1], merge_img.shape[0]), (0, 0, 0), -1) # 设置蓝色为热度图基本色
    cv2.addWeighted(overlay, alpha, merge_img, 1-alpha, 0, merge_img) # 将背景热度图覆盖到原图
    cv2.addWeighted(heatmap_img, alpha, merge_img, 1-alpha, 0, merge_img) # 将热度图覆盖到原图
    cv2.imwrite('%s.overlay.png'%str(pair[0]),merge_img,[int(cv2.IMWRITE_PNG_COMPRESSION),3])
