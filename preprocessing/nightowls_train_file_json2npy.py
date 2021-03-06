import collections
import sys
import os
import json
import pickle
import numpy as np

TRAIN_ANNS = sys.argv[0]
IMAGES_DIR = sys.argv[1]
SAVE_FILE = sys.argv[2]
SAMPLE_TRAIN_FILE = '/home/vobecant/PhD/CSP/data/cache/nightowls/train_h50_nonempty_xyxy'
MIN_HEIGHT = 50
LABELS = [1]
IMH, IMW = 640, 1024

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

    # apply image boundaries
    xyxy[0] = min(IMW - 1, max(0, xyxy[0]))
    xyxy[1] = min(IMH - 1, max(0, xyxy[1]))
    xyxy[2] = min(IMW - 1, max(0, xyxy[2]))
    xyxy[3] = min(IMH - 1, max(0, xyxy[3]))
    return xyxy


'''
Training annotations are list of dictionaries, one per image.
Each dictionary has the following elements:
 - ignoreareas
 - bboxes
 - image_id
 - filepath
'''
ignore = set()

for ann in annotations:
    if ann['ignore']:
        bbox_xywh = ann['bbox']
        bbox_xyxy = xywh2xyxy(bbox_xywh)
        img_ann = choosen_anns[ann['image_id']]
        if 'ignoreareas' in img_ann.keys():
            img_ann['ignoreareas'].append(bbox_xyxy)
        else:
            img_ann['ignoreareas'] = [bbox_xyxy]
        continue

    if ann['category_id'] not in LABELS:
        continue
    bbox_xywh = ann['bbox']
    if bbox_xywh[-1] < MIN_HEIGHT:
        continue
    bbox_xyxy = xywh2xyxy(bbox_xywh)
    img_ann = choosen_anns[ann['image_id']]
    if ann['image_id'] in empty_images.keys(): del empty_images[ann['image_id']]
    img_ann['filepath'] = images_lut[ann['image_id']]
    ig = ann['ignore']
    ignore.add(ig)
    bbox_key = 'ignoreareas' if ig else 'bboxes'
    if bbox_key in img_ann.keys():
        img_ann[bbox_key].append(bbox_xyxy)
    else:
        img_ann[bbox_key] = [bbox_xyxy]

# keep only those images that have some annotated bounding box
# It could have happened that some annotation would have only ignore areas.
vals = choosen_anns.values()
anns_list = []
skipped = 0
for v in vals:
    if 'bboxes' in v.keys():
        v['bboxes'] = np.asarray(v['bboxes'])
        if 'ignoreareas' in v.keys():
            v['ignoreareas'] = np.asarray(v['ignoreareas'])
        else:
            v['ignoreareas'] = np.asarray([])
        anns_list.append(v)
    else:
        skipped += 1
print(
    '{} of images ({} skipped) with some annotated peroson higher than {}'.format(len(anns_list), skipped, MIN_HEIGHT))

with open(SAVE_FILE, 'wb') as f:
    pickle.dump(anns_list, f, protocol=2)
