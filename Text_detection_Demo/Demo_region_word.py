import numpy as np
import cv2
import os
from os.path import join
from Reshape_wb_block import reshape_I
from Init_detection import word_block_analysis
import time
def detection_init(img_path, word_path, block_path, save_path):
	imgs = []
	for i in os.listdir(img_path):
		if '.jpg' in i or 'JPG' in i:
			imgs.append(i)
	imgs.sort()
	nimgs = len(imgs)
	for i in range(nimgs):
		img_name = imgs[i]
		print "Now detect the image is: ", img_name

		img_s = join(img_path, img_name)
		word_s = join(word_path, img_name[:-4] + '_0.40.tif')

		block_s = join(block_path, img_name[:-4] + '_0.40.tif')
		# save_img = join(save_path, img_name[:-4] + '_det.png')
		# save_file = join(save_path, 'res_'+img_name[:-4] + '.txt')
		# txt_file = open(save_file, 'w')

		img = cv2.imread(img_s)
		word = cv2.imread(word_s, 0)
		block = cv2.imread(block_s, 0)

		f = np.zeros((img.shape[0] * img.shape[1], 2))
		word, block = reshape_I(img, word, block)
		# print word.shape, block.shape

		# Analysise word and block
		# word_block_analysis(img, img_name, save_path, word, block, txt_file)
		word_block_analysis(img, img_name, save_path, word, block)

if __name__ == '__main__':
	t1 = time.time()	
	img_path = r'../Train_WSR_TCB/data/Test_dataset/MSRA-TD500'
	word_path = r'../Train_WSR_TCB/data/Test_results/TD_MKEI_Word_MSRA-TD500' 
	block_path = r'../Train_WSR_TCB/data/Test_results/TD_C_blur_Focal_MSRA-TD500'
	save_path = r'./save_detection'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	detection_init(img_path, word_path, block_path, save_path)
	t2 = time.time()
	print "The total time is: ",t2 - t1
print time.asctime( time.localtime(time.time()) )
