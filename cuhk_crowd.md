# CUHK Crowd Dataset

## Description
### a. The whole dataset:

* 474 video clips from 215 crowded scenes.
* Each clip with the extracted trajectories by gKLT tracker is preprocessed by deleting short trajectories, stationary points, and some errors.
* Details of dataset can be found in dataset_info. It contains video name, video length, video size, video source, video t0 (the frame for group detection evaluation), group detection (300 group detection used in our cvpr paper), video_gt (video classes groundtruth), and scene number. (You can also choose any frame to do group detection. The frame list in video_info_t0 is just what we use in our cvpr paper.)
* These data can only be used for academic research purposes.
* The COPYRIGHT of the videos (with watermark) belongs to GettyImages and Pond5 respectively.

### b. Dataset used in paper:

* In our 14'cvpr paper, we take only 30 frames from each clip to implement and evaluate our approach.
* The frame for group detection can be found in dataset_info, and the groundtruth of 300 group detection can be downloaded from here.
* Dataset_info ("video_gt") also provides the groundtruth of video classes corresponding to our defined 8 classes in the cvpr paper.
* You can get 215 independent scene from dataset_info ("scene_number"), different number means different scenes.
* These data can only be used for academic research purposes.



## Image sequences & Trajectories(Baidu disk version)
This is the image version of CUHK crowd dataset. It includes image sequences and KLT tracking results.

If you want videos only, please refer to Video.

There are totally 8 zip files, please find the links as follows:

group_video_clips_imgs.zip.001
Link: http://pan.baidu.com/s/1c0CKgJQ
Password: olvn

group_video_clips_imgs.zip.002
Link: http://pan.baidu.com/s/1dDtLmmd
Password: udje

group_video_clips_imgs.zip.003
Link: http://pan.baidu.com/s/1dDgodfN
Password: qtc1

group_video_clips_imgs.zip.004
Link: http://pan.baidu.com/s/1kTC5NoF
Password: rkyh

group_video_clips_imgs.zip.005
Link: http://pan.baidu.com/s/1ntzA5iL 
Password: uh00

group_video_clips_imgs.zip.006
Link: http://pan.baidu.com/s/1pJAvMob
Password: cqjy

group_video_clips_imgs.zip.007
Link: http://pan.baidu.com/s/14aKto 
Password: bpxg

group_video_clips_imgs.zip.008
Link: http://pan.baidu.com/s/1c0eonsO
Password: 5hcc


Tips: After downloading all 8 files, you can unzip them together with a password: cuhkivp.


##  Video(Baidu disk version)
This is the video version of CUHK crowd dataset. 

If you want the extracted image sequences and trajectories. Please refer to Image sequences & Trajectories.

There are totally 3 zip files, please find the links as follows:

group_video_clips.zip.001
Link: http://pan.baidu.com/s/1qWNlafM 
Password: ljgu

group_video_clips.zip.002
Link: http://pan.baidu.com/s/1gd02GQR 
Password: yczi

group_video_clips.zip.003
Link: http://pan.baidu.com/s/1jGifJjK 
Password: ghqw


Tips: After downloading all three files, you can unzip them together with a password: cuhkivp.



