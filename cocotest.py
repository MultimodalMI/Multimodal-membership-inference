import os
import pickle
import numpy as np
from pycocotools.coco import COCO

train_coco = COCO("data/annotations/captions_train2014.json")
val_coco = COCO("data/annotations/captions_val2014.json")
train_ids = list(train_coco.anns.keys())
val_ids = list(val_coco.anns.keys())
print(len(train_ids))
print(len(val_ids))
for i in range(10):
    annsid = val_ids[i]
    caption = val_coco.anns[annsid]['caption']
    img_id = val_coco.anns[annsid]['image_id']
    path = val_coco.loadImgs(img_id)[0]['file_name']
    print(path)
    print(caption)
for i in range(10):
    annsid = train_ids[i]
    caption = train_coco.anns[annsid]['caption']
    img_id = train_coco.anns[annsid]['image_id']
    path = train_coco.loadImgs(img_id)[0]['file_name']
    print(path)
    print(caption)