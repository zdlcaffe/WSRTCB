# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
from math import sqrt, tan, cos, sin, tan
from skimage import transform, data
from math import pi

# KAIST
def KAIST_guass01(im_png, img, gt_rois):
    Gauss_map = np.zeros((img.shape[0], img.shape[1]))
    Gauss_temp = np.zeros((img.shape[0], img.shape[1]))

    # Tl_h = 0.5
    # Tl_w = 0.95
    Tl_h = 1
    Tl_w = 1

    for i in range(len(gt_rois)):
        print "i is: ", i
        Ax1 = gt_rois[i][0]
        Ay1 = gt_rois[i][1]
        w = int((gt_rois[i][2]- gt_rois[i][0]) * Tl_w)
        h =int((gt_rois[i][3]-gt_rois[i][1])* Tl_h)

        ctr_x = Ax1 + w * 0.5
        ctr_y = Ay1 + h * 0.5

        Gauss_map00 = np.zeros((h, w))
        for x in range(w):
            for y in range(h):
                if w>h:
                    dis = abs(y - 0.5 * h)
                    d = dis*dis

                    det  = h*1.0/6
                    R = (det*det)*0.4    
                                
                    Gauss_map00[y, x] = np.exp(-0.5 * d / R)
                    # dis = abs(y - 0.5 * h)
                    # Gauss_map00[y, x] = np.exp(-0.5 * dis / R)
                else:
                    dis = abs(x - 0.5 * w)
                    d = dis*dis

                    det  = w*1.0/6
                    R = (det*det)*0.4

                    Gauss_map00[y, x] = np.exp(-0.5 * d / R)

        Gauss_temp[int(ctr_y - 0.5 * Gauss_map00.shape[0]):int(ctr_y + 0.5 * Gauss_map00.shape[0]),
        int(ctr_x - 0.5 * Gauss_map00.shape[1]):int(ctr_x + 0.5 * Gauss_map00.shape[1])] = Gauss_map00

        for i in range(Gauss_temp.shape[0]):
            for j in range(Gauss_temp.shape[1]):
                if Gauss_temp[i, j] > 0:
                    Gauss_map[i, j] = Gauss_temp[i, j]
        Gauss_temp = np.zeros((img.shape[0], img.shape[1]))

    plt.imsave(im_png, Gauss_map, cmap=plt.cm.gray)


def endWith(s, *endstring):
    array = map(s.endswith, endstring)
    if True in array:
        return True
    else:
        return False


## KAIST_all
filepath = r"Your data path"
for gt in os.listdir(filepath):
    if endWith(gt, '.txt'):
        gtfile = open(filepath + "\\" + gt, 'r')
        # print gtfile
        gtlines = gtfile.readlines()
        print gtlines
        gtlines_len = len(gtlines)
        imf=filepath + r"\\" + gt[:-4]+".jpg"
        im_png=filepath + r"\\" + gt[:-4]+".png"
        img = cv2.imread(imf)
        gt_rois = []
        gt_gauss_rois = []
        print "imf is:", imf
        for i in range(gtlines_len):
            x1, y1, w, h = gtlines[i].split(" ")[:4]
            gt_gauss_roi = [int(x1), int(y1), int(w), int(h)]
            gt_gauss_rois.insert(-1, gt_gauss_roi)
        KAIST_guass01(im_png, img, gt_gauss_rois)
    else:
        continue
    print "Next"
print "over all"
