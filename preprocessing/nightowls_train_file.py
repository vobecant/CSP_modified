import collections
import sys
import os
import json
import pickle

TRAIN_ANNS = '/home/vobecant/datasets/nightowls/nightowls_training.json'
IMAGES_DIR = '/home/vobecant/datasets/nightowls/nightowls_training'
SAVE_FILE = '/home/vobecant/PhD/CSP/data/cache/nightowls/train_h50_allAnns_xyxy_baseline'
SAMPLE_TRAIN_FILE = '/home/vobecant/PhD/CSP/data/cache/nightowls/train_h50_nonempty_xyxy'
MIN_HEIGHT = 50
LABELS = [1]

with open(SAMPLE_TRAIN_FILE, 'rb') as f:
    sample_anns = pickle.load(f, encoding='latin1')

with open(TRAIN_ANNS, 'r') as f:
    train_anns = json.load(f)

images_lut = {ann['id']: ann['file_name'] for ann in train_anns['images']}
annotations = train_anns['annotations']

choosen_anns = collections.defaultdict(dict)
empty_images = images_lut.copy()


def xywh2xyxy(bbox_xywh):
    xyxy = bbox_xywh.copy()
    xyxy[2] += bbox_xywh[0]
    xyxy[3] += bbox_xywh[1]
    return xyxy


for ann in annotations:
    if ann['category_id'] not in LABELS:
        continue
    bbox_xywh = ann['bbox']
    if bbox_xywh[-1] < MIN_HEIGHT:
        continue
    bbox_xyxy = xywh2xyxy(bbox_xywh)
    img_ann = choosen_anns['image_id']
    if len(img_ann['bbox']) > 0:
        img_ann['bbox'].append(bbox_xyxy)
    else:
        img_ann['bbox'] = [bbox_xyxy]

with open(SAVE_FILE, 'wb') as f:
    pickle.dump(choosen_anns, f, protocol=2)
