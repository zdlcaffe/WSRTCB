# 1. Introduction
This project includes the text detection source code and trained model about the word stroke region and text center block.
# 2. Installation
* Clone the repo  
```
git clone git@github.com:zdlcaffe/WSRTCB.git  
```
* Then you can do as follow:
```
cd ${WSRTCB_root/Train_WSR_TCB/caffe}  
make –j  
make pycaffe 
```
# 3. Testing
## 3.1 Generate WSR/TCB  score map
* Download the [TD_MKEI_Word.caffemodel](https://pan.baidu.com/s/1TNCL6MgWnHBDlFTQFkg5qA), trained on KAIT dataset.  
* Then you can do as follow:
```
cd ${WSRTCB_root/}  
mkdir snapshot  
mkdir pre_model
```
* put TD_MKEI_Word.caffemodel to the fold of ${WSRTCB_root/Train_WSR_TCB/ snapshot}.  

* Suppose you have downloaded the MSRA-TD500 dataset, execute the following commands to test the model on MSRA-TD500.  Then you can do as follow:
```
cd ${WSRTCB_root/Train_WSR_TCB/demo}  
python Demo.py  
```
## 3.2 There are some samples:  

![image](https://github.com/zdlcaffe/WSRTCB/blob/master/Train_WSR_TCB/data/Test_results/TD_MKEI_Word_MSRA-TD500/IMG_0059.png)
![image]( https://github.com/zdlcaffe/WSRTCB/blob/master/Train_WSR_TCB/data/Test_results/TD_MKEI_Word_MSRA-TD500/IMG_0616.png)
![image]( https://github.com/zdlcaffe/WSRTCB/blob/master/Train_WSR_TCB/data/Test_results/TD_MKEI_Word_MSRA-TD500/IMG_1846.png)
![image](https://github.com/zdlcaffe/WSRTCB/blob/master/Train_WSR_TCB/data/Test_results/TD_MKEI_Word_MSRA-TD500/IMG_2257.png) 

## 3.3 Threshold WSR/TCB maps:
You can do as follow:
```
cd ${WSRTCB_root/Text_Demo}  
python fuse_thred	.py  
```
## 3.4 Generate detection results  
You can do as follow:
```
cd ${WSRTCB_root/Text_Demo}  
python Demo_region_word.py
```
## 3.5 There are some samples:

![image](https://github.com/zdlcaffe/WSRTCB/blob/master/Text_detection_Demo/save_detection/det_result/IMG_0059_det.jpg)
![image](https://github.com/zdlcaffe/WSRTCB/blob/master/Text_detection_Demo/save_detection/det_result/IMG_0616_det.jpg)
![image](https://github.com/zdlcaffe/WSRTCB/blob/master/Text_detection_Demo/save_detection/det_result/IMG_1846_det.jpg)
![image](https://github.com/zdlcaffe/WSRTCB/blob/master/Text_detection_Demo/save_detection/det_result/IMG_2257_det.jpg)

# 4. Training
Download the pretrained model [vgg16convs.caffemodel](https://pan.baidu.com/s/1IEt48THcdmncH2zoeokypA), and put it to 
${WSRTCB_root/Train_WSR_TCB/pre_model}

## 4.1 Generate your map  
Scripts for generating ground truth have been provided in the label_generate directory. It not hard to write a converting script for your own dataset.

## 4.2 Train your own model
Modify ${WSRTCB_root/Train_WSR_TCB/model/TD_MKEI_Word.py} to configure your dataset name and dataset path like:  
......  
data_params['root'] = 'data/MKEIWord'  
data_params['source'] = "MKEI_Word.lst"  
......

## 4.3 Start training

You can do as follow:
```
cd ${WSRTCB_root/Train_WSR_TCB}  
sh ./train.sh 
```
## Citation
Use this bibtex to cite this repository:
```
@article{liu2019scene,
  title={Scene text detection with fully convolutional neural networks},
  author={Liu, Zhandong and Zhou, Wengang and Li, Houqiang},
  journal={Multimedia Tools and Applications},
  pages={1--23},
  year={2019},
  publisher={Springer}
}
```
# Acknowlegement
