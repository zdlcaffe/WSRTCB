# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
from math import sqrt, tan, cos, sin, tan
from skimage import transform, data
from math import pi


def endWith(s, *endstring):
    array = map(s.endswith, endstring)
    if True in array:
        return True
    else:
        return False

# ICDAR 2013
def heri_vers_rectangle(im_png, img, gt_rois):
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
        Ax3 = gt_rois[i][2]
        Ay3 = gt_rois[i][3]
        Ax2 = Ax1;
        Ay2 = Ay3
        Ax4 = Ax3;
        Ay4 = Ay1
        Ax1 = int(Ax1);
        Ax2 = int(Ax2);
        Ax3 = int(Ax3);
        Ax4 = int(Ax4)
        Ay1 = int(Ay1);
        Ay2 = int(Ay2);
        Ay3 = int(Ay3);
        Ay4 = int(Ay4)

        w = int((Ax3 - Ax1) * Tl_w);
        h = int((Ay3 - Ay1) * Tl_h)

        ctr_x = (Ax1 + Ax3) * 0.5
        ctr_y = (Ay1 + Ay3) * 0.5
        # R = 0.25 * h
        # R = 0.5 * h
        if w > h: 
            det  = h*1.0/6
            R = det*det

            Gauss_map00 = np.zeros((h, w))
            for x in range(w):
                for y in range(h):
                    # dis = abs(y - 0.5 * h)
                    # Gauss_map00[y, x] = np.exp(-0.5 * dis / R)

                    dis = abs(y - 0.5 * h)
                    d = dis*dis
                    Gauss_map00[y, x] = np.exp(-0.5 * d / R)


            Gauss_temp[int(ctr_y - 0.5 * Gauss_map00.shape[0]):int(ctr_y + 0.5 * Gauss_map00.shape[0]),
            int(ctr_x - 0.5 * Gauss_map00.shape[1]):int(ctr_x + 0.5 * Gauss_map00.shape[1])] = Gauss_map00

            for i in range(Gauss_temp.shape[0]):
                for j in range(Gauss_temp.shape[1]):
                    if Gauss_temp[i, j] > 0:
                        Gauss_map[i, j] = Gauss_temp[i, j]
        else:
            w0 = int((Ax3 - Ax1) * Tl_w);
            h0 = int((Ay3 - Ay1) * Tl_h)
            w = h0
            h = w0

            det  = h*1.0/6
            R = det*det
            Gauss_map00 = np.zeros((w, h))
            for x in range(h):
                for y in range(w):
                    # dis = abs(y - 0.5 * h)
                    # Gauss_map00[y, x] = np.exp(-0.5 * dis / R)

                    dis = abs(x - 0.5 * h)
                    d = dis*dis
                    Gauss_map00[y, x] = np.exp(-0.5 * d / R)


            Gauss_temp[int(ctr_y - 0.5 * Gauss_map00.shape[0]):int(ctr_y + 0.5 * Gauss_map00.shape[0]),
            int(ctr_x - 0.5 * Gauss_map00.shape[1]):int(ctr_x + 0.5 * Gauss_map00.shape[1])] = Gauss_map00

            for i in range(Gauss_temp.shape[0]):
                for j in range(Gauss_temp.shape[1]):
                    if Gauss_temp[i, j] > 0:
                        Gauss_map[i, j] = Gauss_temp[i, j]        



        Gauss_temp = np.zeros((img.shape[0], img.shape[1]))

    plt.imsave(im_png, Gauss_map, cmap=plt.cm.gray)

##*******************************************************#
## ICDAR2013
filepath=r"Your data path"
for gt in os.listdir(filepath):
    if endWith(gt, '.txt'):
        gtfile = open(filepath + "\\" + gt, 'r')
        # print gtfile
        gtlines = gtfile.readlines()
        gtlines_len = len(gtlines)
        # imf=filepath + r"\\" +gt[3:][:-4]+".jpg"
        # im_png=filepath + r"\\" +gt[3:][:-4]+".png"
        imf=filepath + r"\\" +gt[:-4]+".jpg"
        im_png=filepath + r"\\" +gt[:-4]+".png"
        img = cv2.imread(imf)
        gt_rois=[]
        for i in range(gtlines_len):
            x1, y1, x3, y3= gtlines[i].split(" ")[:4]
            gt_roi =[int(x1),int(y1),int(x3),int(y3)]
                # x0, y0, w, h, theta= gtlines[i].split(" ")
                # gt_roi =[int(x0),int(y0),int(w),int(h),float(theta)]

            gt_rois.insert(-1,gt_roi)
            # print gt_rois
        heri_vers_rectangle(im_png, img, gt_rois)
    else:
        continue
    print "Next"
print "over all"
