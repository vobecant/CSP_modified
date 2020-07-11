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

images_lut = {ann['id']: os.path.join(IMAGES_DIR, ann['file_name']) for ann in train_anns['images']}
annotations = train_anns['annotations']

choosen_anns = collections.defaultdict(dict)
empty_images = images_lut.copy()


def xywh2xyxy(bbox_xywh):
    xyxy = bbox_xywh.copy()
    xyxy[2] += bbox_xywh[0]
    xyxy[3] += bbox_xywh[1]
    return xyxy


'''
Training annotations are list of dictionaries, one per image.
Each dictionary has the following elements:
 - ignoreareas
 - bboxes
 - image_id
 - filepath
'''

for ann in annotations:
    if ann['category_id'] not in LABELS:
        continue
    bbox_xywh = ann['bbox']
    if bbox_xywh[-1] < MIN_HEIGHT:
        continue
    bbox_xyxy = xywh2xyxy(bbox_xywh)
    img_ann = choosen_anns[ann['image_id']]
    if ann['image_id'] in empty_images.keys(): del empty_images[ann['image_id']]
    img_ann['filepath'] = images_lut[ann['image_id']]
    bbox_key = 'ignoreareas' if ann['ignore'] else 'bbox'
    if bbox_key in img_ann.keys():
        img_ann[bbox_key].append(bbox_xyxy)
    else:
        img_ann[bbox_key] = [bbox_xyxy]

with open(SAVE_FILE, 'wb') as f:
    pickle.dump(choosen_anns, f, protocol=2)
