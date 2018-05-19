# -*- coding: utf-8 -*-
import numpy as np
import scipy.misc
import cv2
import scipy.io
import os, sys, argparse
from os.path import join, splitext, split, isfile
import time 
import matplotlib
import cv2
import math
from math import sqrt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '../caffe/python')
import caffe

def forward(data):
  assert data.ndim == 3
  data -= np.array((104.00698793,116.66876762,122.67891434))
  data = data.transpose((2, 0, 1))
  net.blobs['data'].reshape(1, *data.shape)
  net.blobs['data'].data[...] = data
  return net.forward()

parser = argparse.ArgumentParser(description='Forward all testing images.')

parser.add_argument('--model', type=str, default='../snapshot/TD_MKEI_Word.caffemodel')
parser.add_argument('--net', type=str, default='../model/TD_test.pt')

# parser.add_argument('--output', type=str, default='sigmoid_dsn3') # output field
# parser.add_argument('--output', type=str, default='sigmoid_dsn4') # output field
# parser.add_argument('--output', type=str, default='sigmoid_dsn5') # output field
parser.add_argument('--output', type=str, default='sigmoid_fuse') # output field
parser.add_argument('--gpu', type=int, default= 1)
parser.add_argument('--ms', type=bool, default=False) # Using multiscale
parser.add_argument('--savemat', type=bool, default=False) # whether save .mat
args = parser.parse_args()


assert isfile(args.model) and isfile(args.net), 'file not exists'
caffe.set_mode_gpu()
caffe.set_device(args.gpu)

net = caffe.Net(args.net, args.model, caffe.TEST)

test_dir = '../data/Test_dataset/MSRA-TD500/'
test_dir_lst = test_dir.split('/')

# multi-scales
if args.ms:
  sfs = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0,2.2, 2.4]
else:
  sfs = [1.0]

t1 = time.time()
for sf in sfs:
  scales01 =[0.4, 0.8, 1, 1.4, 1.6]  
  if args.ms:
    save_dir = join('../data/Test_result/', splitext(split(args.model)[1])[0]) # directory to save results
    save_dir = save_dir + '_ms_'+str(sf)+test_dir_lst[len(test_dir_lst)-2]
  else:
    save_dir = join('../data/Test_result/', splitext(split(args.model)[1])[0]+'_'+test_dir_lst[len(test_dir_lst)-2]) 

  if not os.path.exists(save_dir):
      os.makedirs(save_dir)
  imgs = [i for i in os.listdir(test_dir) if '.jpg' in i or '.JPG' in i]
  imgs.sort()

  nimgs = len(imgs)
  print "totally "+str(nimgs)+"images"

  scale = 1228800
  max_side = 1080
  scale01 = 960

  for i in range(nimgs):
    img = imgs[i]
    print img
    img = cv2.imread(join(test_dir, img))
    h, w, _ = img.shape
    ori_h, ori_w, _ = img.shape 
    print img.shape
    if h*w>=scale and h>w:
      img = cv2.resize(img, (int(img.shape[1] * max_side / img.shape[0]), max_side))
    elif h*w>=scale and h<w:
      img = cv2.resize(img, (max_side, int(img.shape[0] * max_side / img.shape[1])))

    elif h*w<scale and h<w and h>scale01 and w < max_side:
      img = cv2.resize(img, (max_side, int(img.shape[0] * max_side / img.shape[1])))
    elif h*w<scale and h>w and w>scale01 and h < max_side:
      img = cv2.resize(img, (int(img.shape[1] * max_side / img.shape[0]), max_side))

    elif h<=scale01 and h>w:
      img = cv2.resize(img, (int(img.shape[1] * scale01 / img.shape[0]), scale01))
    elif w<=scale01 and h<w:
      img = cv2.resize(img, (scale01, int(img.shape[0] * scale01 / img.shape[1])))
    else:
      pass 
    print img.shape
    h0, w0, _= img.shape
    img = cv2.resize(img , (int(w0/sf), h0))
    print img.shape
    h1, w1, _=img.shape
    delta = w1*1.0/h1
    h3 = int(sqrt(scale/delta))
    w3 = int(delta*h3)
    img = cv2.resize(img , (w3, h3))
    print img.shape

    if img.ndim == 2:
      img = img[:, :, np.newaxis]
      img = np.repeat(img, 3, 2)
    h, w, _ = img.shape
    edge = np.zeros((h, w), np.float32)
    if args.ms:
      scales = scales01    
    else:
      scales = [1]
      
    for s in scales:
      h1, w1 = int(s * h), int(s * w)
      img1 = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_CUBIC).astype(np.float32)
      edge1 = np.squeeze(forward(img1)[args.output][0, 0, :, :])
      edge += cv2.resize(edge1, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    edge /= len(scales)
    fn, ext = splitext(imgs[i])
    print fn, ext

    ori_image = cv2.resize(edge,(ori_w, ori_h), interpolation=cv2.INTER_AREA)
    plt.imsave(join(save_dir, fn+'.png'), ori_image)

    print "Saving to '" + join(save_dir, imgs[i][0:-4]) + "', Processing %d of %d..."%(i + 1, nimgs)
    print '*************************'

t2 = time.time()
print time.asctime( time.localtime(time.time()))
print "The total time is: ",t2 - t1