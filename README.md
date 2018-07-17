# CIDNN 
CIDNN: Encoding Crowd Interaction with Deep Neural Network 

This repo is the official open source of CIDNN, CVPR 2018 by Yanyu Xu, Zhixin Piao and Shenghua Gao. 

![architecture](img/architecture.png)

It is implemented in Pytorch and Python 2.7.x.

If you find this useful, please cite our work as follows:

```
@INPROCEEDINGS{xu2018cidnn, 
	author={Yanyu Xu and Zhixin Piao and Shenghua Gao}, 
	booktitle={2018 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
	title={Encoding Crowd Interaction with Deep Neural Network for Pedestrian Trajectory Prediction}, 
	year={2018}
}
```

## DataSet

|      DataSet       |                             Link                             |
| :----------------: | :----------------------------------------------------------: |
|       GC [1]       | [BaiduYun](https://pan.baidu.com/s/1dD0EmXF) or [DropBox](https://www.dropbox.com/s/7y90xsxq0l0yv8d/cvpr2015_pedestrianWalkingPathDataset.rar?dl=0) |
|      ETH [2]       | [website](http://www.vision.ee.ethz.ch/~stefpell/lta/index.html) |
|      UCY [3]       | [website](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data) |
|   CUHK Crowd [4]   | [website](http://www.ee.cuhk.edu.hk/~jshao/projects/CUHKcrowd_files/cuhk_crowd_dataset.htm) |
| subway station [5] | [website](http://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html) |

**Update 2019.07.17**: Because the website of GC Dataset has been deleted, we give the download link and decription of it.

**GC Dataset Description:**

```reStructuredText
1. This dataset contains two folders, naming ‘Annotation’ and ‘Frame’, respectively.

2. The ‘Annotation’ folder contains the manually labeled walking paths of 12,684 pedestrians. Annotations are named as ‘XXXXXX.txt’. ‘XXXXXX’ is pedestrian index.

3. For each of the annotation txt file. It contains multiple integers, corresponding to the (x,y,t)s of the current pedestrian. ‘x’ and ‘y’ are point coordinates and ‘t’ is frame index. There should be 3N integers if this pedestrian appears in N frames. All pedestrians within Frame 000000 to 100000 are labeled from the time point he(she) arrives to the time point he(she) leaves.

4. The ‘Frame’ folder contains 6001 frames sampled from a surveillance video captured at the Grand Central Train Station of New York. These frames are named as ‘XXXXXX.jpg’. ‘XXXXXX’ is frame index. It starts from ‘000000’ and ends at ‘120000’. One frame is sampled every 20 frames from the surveillance video clip.
```



## Reference

1. **Understanding Pedestrian Behaviors from Stationary Crowd Groups**

   Shuai Yi, Hongsheng Li, and Xiaogang Wang.  In CVPR, 2015.

2. **You’ll never walk alone: Modeling social behavior for multi-target tracking**

   Stefano Pellegrini, Andreas Ess, Konrad Schindler, Luc Van Gool. In ICCV 2009.

3. **Crowds by Example**

   Alon Lerner, Yiorgos Chrysanthou, Dani Lischinski. In EUROGRAPHICS 2007.

4. **Scene-Independent Group Profiling in Crowd**

   Jing Shao, Chen Change Loy, Xiaogang Wang. In CVPR 2014.

5. **Understanding Collective Crowd Behaviors: Learning a Mixture Model of Dynamic Pedestrian-Agents**

   Bolei Zhou, Xiaogang Wang, Xiaoou Tang. In CVPR 2012.

