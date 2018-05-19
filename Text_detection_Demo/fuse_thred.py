# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from os.path import join
import copy
import matplotlib.pyplot as plt
import time
rootdir = r"../Train_WSR_TCB/data/Test_results"
list = os.listdir(rootdir)
list.sort()

print "There are ", len(list),"flods"
N = 0
s1=[]
s2=[]

for line in list:
	N = N + 1
	img_path = os.path.join(rootdir,line)
	if os.path.isdir(img_path):
		print "This is "+str(N) + " flod: " + img_path		
		save_path = img_path
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		imgs=[]
		for i in os.listdir(img_path):
		  if 'png' in i or 'jpg' in i:
			imgs.append(i)
		imgs.sort()
		nimgs = len(imgs)
		for i in range(nimgs):			
			img_name =imgs[i]
			img = join(img_path, img_name)
			I = cv2.imread(img, 0)
			th01 = np.mean(I)
			th02 = np.median(I)
			th03 = np.max(I)
			th04 = np.min(I)
			th05 = I.min()
			th06 = I.max()

			ratios = [0.30,0.32,0.34,0.36,0.38,0.4,0.45,0.5]
			s1.append(th01)
			s2.append(th02)
			for ratio in ratios:
				I2 = np.zeros((I.shape[0], I.shape[1]))
				if ratio in [0.30,0.40,0.50]:
					save_img =join(save_path, img_name[:-4]+'_'+str(ratio)+'0.tif')
				else:
					save_img =join(save_path, img_name[:-4]+'_'+str(ratio)+'.tif')
					
				ret, binary = cv2.threshold(I,int(ratio*255),255,cv2.THRESH_BINARY) 
				cv2.imwrite(save_img, binary)
	else:
		print "There is no flod"
print time.asctime(time.localtime(time.time()))