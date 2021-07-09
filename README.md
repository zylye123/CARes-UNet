
# CARes-UNet

CARes-UNet: Content-Aware Residual UNet for Lesion Segmentation of COVID-19 from Chest CT Images

## 0. Table of Contents
- [CARes-UNet](#cares-unet)
  * [0. Table of Contents](#0-table-of-contents)
  * [1. Authors & Maintainers](#1-authors---maintainers)
  * [2. Change Log](#2-change-log)
  * [3. Introduction](#3-introduction)
  * [4. Method](#4-method)
    + [4.1 Network Architecture](#41-network-architecture)
    + [4.2 Semi-supervised Framework](#42-semi-supervised-framework)
  * [5. Experiment](#5-experiment)
    + [5.1 Prerequisites](#51-prerequisites)
    + [5.2 Data Preparation](#52-data-preparation)
    + [5.3 Training](#53-training)
    + [5.4 Testing](#54-testing)
  * [6. License](#6-license)
  * [7. To-Do List](#7-to-do-list)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## 1. Authors & Maintainers

- [Yi Zhang|@zylye123](https://github.com/zylye123)
- [Zixuan Tang|@sysu19351118](https://github.com/sysu19351118)
- [Xinhua Xu|@sysu19351158](https://github.com/sysu19351158)
- [Youjun Zhao|@zhaoyjoy](https://github.com/zhaoyjoy)
- [Yuhang Wen|@Necolizer](https://github.com/Necolizer)

## 2. Change Log

- [2021/06/26] Improve Readme reading experience and provide more details
- [2021/03/27] Modify Readme file.
- [2021/03/26] Create repository and release source code. 

## 3. Introduction

*Coronavirus Disease 2019* (COVID-19) has caused a serious global health crisis. It has been proven that the deep learning method has great potential to assist doctors in diagnosing COVID-19 by **automatically segmenting the lesions in computed tomography (CT) slices**. However, there are still several challenges including high variation in lesion characteristics and low-contrast between lesion areas and healthy tissues. What’s more, lack of high-quality labeled samples and large number of patients lead to the urgency to develop a high accuracy model which performs well not only under supervision but also with semi-supervised methods. 

The main contributions of our work can be summarized as follows:  

1. To develop high accuracy supervised and semi-supervised lesion segmentation deep learning model to assist doctors in diagnosing COVID-19, we propose a novel **content-aware Residual Network (CARes-UNet)** to segment the lesion areas from the chest CT slices. Based on UNet, we added the residual connection to the convolution block to address the degradation problem. We improved the lesion localization accuracy while reducing the computation cost by applying a content-aware upsampling module. Also, we introduced a training technique - using a Ranger optimizer to stable optimizing and to achieve better performance as well as faster convergence. 
2. To alleviate the lacking of high-quality annotated samples, we further developed **a semi-supervised framework** to train the model.
3. We evaluated our approach using a public dataset. We conducted several comparisons and ablation studies to prove the effectiveness of our proposed methods. In our experiments, **our method outperforms other models in multiple indicators**, for instance in terms of dice, CARes-UNet got the score 0.731 and semi-CARes-UNet further boosted it to 0.776.  

## 4. Method

### 4.1 Network Architecture

![](img/Fig1_Network_Architecture.jpg)

Figure 1: Network architecture of CARes-UNet. CARes-UNet comprises a down-sampling path and an up-sampling path. The down-sampling path enlarged the receptive field while reducing computation cost. The up-sampling path recovered the lost resolution in the down-sampling path. Arrows of different colors present different operations.

### 4.2 Semi-supervised Framework

![](img/Fig2_Semi-supervised_Framework.jpg)

Figure 2: Overview of our semi-supervised framework. Pretrained CARes-UNet generates pseudo labels for 1600 images without gold standard, then treats these pseudo labels as masks of the unlabeled to train semi-CARes-UNet together with the labeled ones.

## 5. Experiment

### 5.1 Prerequisites

The code is built at least with the following libraries:

- [Python](https://www.python.org/)
- [Anaconda](https://www.anaconda.com/)
- [PyTorch](https://pytorch.org/)
- Python Imaging Library
- matplotlib

### 5.2 Data Preparation

1. Download the [COVID-SemiSeg Dataset](https://github.com/DengPingFan/Inf-Net)
2. Preprocess
3. Divide all the labeled ones into training set and testing set
4. Put labeled ones and unlabeled ones into the following repository (which should be created firstly):
   - `.\Train-Image`: origin CT slices of labeled training set
   - `.\Train-Mask` : ground truth of labeled training set
   - `.\Test-Image` : origin CT slices of labeled testing set
   - `.\Test-Mask` : ground truth of labeled testing set
   - `.\Unlabeled` : unlabeled  CT slices

### 5.3 Training

- For supervised training, run

```bash
python supervised_train.py
```

- For semi-supervised training, run

```bash
python supervised_train.py
python fake_labels_generator.py --model_path path\to\model --Image_dir .\Unlabeled --Mask-dir .\Pos-Mask
python semi-supervised_train.py --semi_img_path .\Unlabeled
```

`path\to\model ` refers to the path of checkpoint saved in supervised training.

### 5.4 Testing

For testing, run

```bash
python test.py --pretrained_model path\to\model
```

`path\to\model ` refers to the path of checkpoint saved in supervised or semi-supervised training.

Lesion Segmentation results will be stored in `.\CARes_Unet`.

## 6. License

[MIT](https://github.com/zylye123/CARes-UNet/blob/master/LICENSE)

## 7. To-Do List

- [ ] Paper
- [x] Create GUI for presentation
- [ ] Improve semi-supervised learning methods
- [ ] Test on a larger dataset
