# M4I
The experimental code of NIPS 2022 paper "M<sup>4</sup>I: Multi-modal Models Membership Inference".

## Introduction
This repository contains the code that implements the M<sup>4</sup>I described in our paper "M<sup>4</sup>I: Multi-modal Models Membership Inference" published at NIPS 2022. This work studies the privacy leakage of multi-modal models through the lens of membership inference attack, a process of determining whether a data record involves in the model training process or not.

## Requrements
Our code is implemented and tested on Pytorch with the other packages in requirements.txt, you can quickly establish your environment by anaconda through:
`conda create --name 'M4I' --file requirements.txt`
`conda activate 'M4I'`

## Dataset
Please download Flickr8K[<sup>1</sup>](#refer-anchor) and coco2017[<sup>2</sup>](#refer-anchor) from public sources and put them under data folder. 

These datasets can be accessed from following link:

- https://www.kaggle.com/datasets/adityajn105/flickr8k
- https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset

<div id = "refer-anchor"></div>
[1] From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. *Transactions of the Association for Computational Linguistics, 2:67–78, 2014* 

[2] Microsoft coco: Common objects in context. *ECCV 2014*

## Training models
Please following the instructions in our paper to separate the datasets into member data, shadow data, member data, and combination set through random sampling. And train the target model, shadow model and MFE model through:
`python train.py`
`python MFE_train.py`

## Membership inference attack
The script `mi_attack.py` can be used to implement metric-based membership inference and feature-based membership inference.




