# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
from math import sqrt, tan, cos, sin, tan
from skimage import transform, data
from math import pi

#  MSRA-TD500, HUST-TR400, USTB-SV1K_V1
def mul_vertices_rectangle(im_png, img, gt_rois):
    Gauss_map = np.zeros((img.shape[0], img.shape[1]))
    Gauss_temp = np.zeros((img.shape[0], img.shape[1]))
    print Gauss_temp.shape

    Tl_h = 1
    Tl_w = 1

    for i in range(len(gt_rois)):
        print "i is: ", i
        ctr_x = gt_rois[i][0] + gt_rois[i][2] * 0.5
        ctr_y = gt_rois[i][1] + gt_rois[i][3] * 0.5
        # x0=gt_rois[i][0]; y0=gt_rois[i][1]
        Atheta = gt_rois[i][4]

        if gt_rois[i][2] > gt_rois[i][3]:
            w = int(gt_rois[i][2] * Tl_w)
            h = int(gt_rois[i][3] * Tl_h)

            # R = 0.25 * h
            det  = h*1.0/6
            R = det*det*0.4

            Gauss_map00 = np.zeros((h, w))

            for x in range(w):
                for y in range(h):
                    if x > img.shape[0]:
                        x = img.shape[0]
                    elif y >img.shape[1]:
                        y = img.shape[1]
                    else:
                        pass
                    dis = abs(y - 0.5 * h)
                    d = dis*dis
                    # Gauss_map00[y, x] = np.exp(-0.5 * dis / R)
                    Gauss_map00[y, x] = np.exp(-0.5 * d / R)
            rotate_Gmap = transform.rotate(Gauss_map00, -Atheta * 180 / pi, resize=True)
            # cv2.imshow('a',rotate_Gmap)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            T = int(ctr_y - 0.5 * rotate_Gmap.shape[0])
            B = int(ctr_y + 0.5 * rotate_Gmap.shape[0])
            L = int(ctr_x - 0.5 * rotate_Gmap.shape[1])
            R = int(ctr_x + 0.5 * rotate_Gmap.shape[1])
    
            w0 = rotate_Gmap.shape[1]
            h0 = rotate_Gmap.shape[0]

            w1 = Gauss_temp.shape[1]
            h1 = Gauss_temp.shape[0]

            if L<=0:
                L = 2
                Gauss_temp[T:B,int(0.5*w0-ctr_x): int(0.5*w0+ctr_x)] = rotate_Gmap[:, int(0.5*w0-ctr_x): int(0.5*w0+ctr_x)]
            elif T<=0:
                T = 2
                Gauss_temp[int(0.5*h0 - ctr_y):int(0.5*h0 +ctr_y),L:R] = rotate_Gmap[int(0.5*h0 - ctr_y):int(0.5*h0 +ctr_y), :]

            elif R > Gauss_temp.shape[1]:
                R = Gauss_temp.shape[1]
                # print "T B R L is: ", T," ",B, " ",R," ",L," ", int(ctr_x -0.5*w0)," ",int(w1-ctr_x+0.5*w0)
                Gauss_temp[T:B, int(ctr_x -0.5*w0):R] = rotate_Gmap[:, 0:int(w1-ctr_x+0.5*w0)]

            elif  B>Gauss_temp.shape[0]:                            
                B =Gauss_temp.shape[0]
                Gauss_temp[int(ctr_y-0.5*h0):B,L:R] = rotate_Gmap[0:int(B-ctr_y+0.5*h0), :]
            else:
                Gauss_temp[T:B,L:R] = rotate_Gmap            
            

            for i in range(Gauss_temp.shape[0]):
                for j in range(Gauss_temp.shape[1]):
                    if Gauss_temp[i, j] > 0:
                        Gauss_map[i, j] = Gauss_temp[i, j]
                    else:
                        pass
        else:

            w = int(gt_rois[i][2] * Tl_h)
            h = int(gt_rois[i][3] * Tl_w)

            # R = 0.5 * w
            # R ==0.5*w == 3*det
            det  = w*1.0/6
            R = det*det*0.4

            Gauss_map00 = np.zeros((h, w))

            for y in range(h):
                for x in range(w):
                    dis = abs(x - 0.5 * w)
                    # Gauss_map00[y, x] = np.exp(-0.5 * dis / R)

                    d = dis*dis
                    # Gauss_map00[y, x] = np.exp(-0.5 * dis / R)
                    Gauss_map00[y, x] = np.exp(-0.5 * d / R)
           
            rotate_Gmap = transform.rotate(Gauss_map00, -Atheta * 180 / pi, resize=True)
            # cv2.imshow('a',rotate_Gmap)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            T = int(ctr_y - 0.5 * rotate_Gmap.shape[0])
            B = int(ctr_y + 0.5 * rotate_Gmap.shape[0])
            L = int(ctr_x - 0.5 * rotate_Gmap.shape[1])
            R = int(ctr_x + 0.5 * rotate_Gmap.shape[1])
    
            w0 = rotate_Gmap.shape[1]
            h0 = rotate_Gmap.shape[0]

            w1 = Gauss_temp.shape[1]
            h1 = Gauss_temp.shape[0]

            if L<=0:
                L = 2
                Gauss_temp[T:B,int(0.5*w0-ctr_x): int(0.5*w0+ctr_x)] = rotate_Gmap[:, int(0.5*w0-ctr_x): int(0.5*w0+ctr_x)]
            elif T<=0:
                T = 2
                Gauss_temp[int(0.5*h0 - ctr_y):int(0.5*h0 +ctr_y),L:R] = rotate_Gmap[int(0.5*h0 - ctr_y):int(0.5*h0 +ctr_y), :]

            elif R > Gauss_temp.shape[1]:
                R = Gauss_temp.shape[1]-1
                # print "T B R L is: ", T," ",B, " ",R," ",L," ", int(ctr_x -0.5*w0)," ",int(w1-ctr_x+0.5*w0)
                Gauss_temp[T:B, int(ctr_x -0.5*w0):R] = rotate_Gmap[:, 0:int(w1-ctr_x+0.5*w0)]

            elif  B>Gauss_temp.shape[0]:                            
                B =Gauss_temp.shape[0]
                Gauss_temp[int(ctr_y-0.5*h0):B,L:R] = rotate_Gmap[0:int(h1-ctr_y+0.5*h0), :]
            else:
                Gauss_temp[T:B,L:R] = rotate_Gmap    

            for i in range(Gauss_temp.shape[0]):
                for j in range(Gauss_temp.shape[1]):
                    if Gauss_temp[i, j] > 0:
                        Gauss_map[i, j] = Gauss_temp[i, j]
                    else:
                        pass

        Gauss_temp = np.zeros((img.shape[0], img.shape[1]))
    plt.imsave(im_png, Gauss_map, cmap=plt.cm.gray)


def endWith(s, *endstring):
    array = map(s.endswith, endstring)
    if True in array:
        return True
    else:
        return False


# ##********************************************#
# # MSRA-TD500, USTB-SV1K_V1, HUST-TR400
filepath=r"Your data path"
for gt in os.listdir(filepath):
    if endWith(gt, '.gt'):
        gtfile = open(filepath + "\\" + gt, 'r')
        # print gtfile
        gtlines = gtfile.readlines()
        print gtlines
        gtlines_len = len(gtlines)
        imf=filepath + r"\\" + gt[:-3]+".jpg"
        im_png=filepath + r"\\" + gt[:-3]+".png"
        img = plt.imread(imf)
        print imf
        gt_rois=[]
        for i in range(gtlines_len):
            s0, s1, x0, y0, w, h, theta= gtlines[i].split(" ")[:7]
            gt_roi =[int(x0),int(y0),int(w),int(h),float(theta)]
            gt_rois.insert(-1,gt_roi)
        mul_vertices_rectangle(im_png, img, gt_rois)
    else:
        continue
    print "Next"
print "over all"