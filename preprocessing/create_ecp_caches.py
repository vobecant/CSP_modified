import json
import pickle
import os
import sys
from pathlib import Path
import numpy as np

DATASET_DIR = sys.argv[1]  # /home/vobecant/datasets/ecp/ECP
SAVE_DIR = sys.argv[2]  # /home/vobecant/datasets/ecp/ECP
TRN_ANN_DIR = os.path.join(DATASET_DIR, 'day', 'labels', 'train')
VAL_ANN_DIR = os.path.join(DATASET_DIR, 'day', 'labels', 'val')

trn_files = [os.path.join(TRN_ANN_DIR, f) for f in Path(TRN_ANN_DIR).rglob('*.json')]
val_files = [os.path.join(VAL_ANN_DIR, f) for f in Path(VAL_ANN_DIR).rglob('*.json')]

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
save_file_trn = os.path.join(SAVE_DIR, 'train_h50')
save_file_val = os.path.join(SAVE_DIR, 'val_h50')

'''
Annotations for CSP training are in the form of list of dictionaries. Each such dict contains:
'ignoreareas'
'vis_bboxes'
'bboxes'
'filepath'
'''

# TODO: train files
anns_list = []
for trnf in trn_files:
    with open(trnf, 'r') as f:
        trn = json.load(f)

    img_path = trnf.replace('.json', '.png').replace('/labels/', '/img/')
    assert os.path.exists(img_path)

    ann = {'ignoreareas': [],
           'vis_bboxes': np.asarray([]),
           'bboxes': [],
           'filepath': img_path}

    for ch in trn['children']:
        bbox = [ch['x0'], ch['y0'], ch['x1'], ch['y1']]
        height = bbox[-1] - bbox[1]
        if ch['identity'] == 'pedestrian' and height >= 50:
            ann['bboxes'].append(bbox)
        else:
            ann['ignoreareas'].append(bbox)

    ann['bboxes'] = np.asarray(ann['bboxes'])
    ann['ignoreareas'] = np.asarray(ann['ignoreareas'])

    anns_list.append(ann)

with open(save_file_trn, 'rb') as f:
    pickle.dump(anns_list, f)

# TODO: validation files
anns_list = []
for trnf in val_files:
    with open(trnf, 'r') as f:
        trn = json.load(f)

    img_path = trnf.replace('.json', '.png').replace('/labels/', '/img/')
    assert os.path.exists(img_path)

    ann = {'ignoreareas': [],
           'vis_bboxes': np.asarray([]),
           'bboxes': [],
           'filepath': img_path}

    for ch in trn['children']:
        bbox = [ch['x0'], ch['y0'], ch['x1'], ch['y1']]
        height = bbox[-1] - bbox[1]
        if ch['identity'] == 'pedestrian' and height >= 50:
            ann['bboxes'].append(bbox)
        else:
            ann['ignoreareas'].append(bbox)

    ann['bboxes'] = np.asarray(ann['bboxes'])
    ann['ignoreareas'] = np.asarray(ann['ignoreareas'])

    anns_list.append(ann)

with open(save_file_val, 'rb') as f:
    pickle.dump(anns_list, f)
