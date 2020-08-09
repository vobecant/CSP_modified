import json
import pickle
import os
import sys
from pathlib import Path
import numpy as np

DATASET_DIR = sys.argv[1]  # /home/vobecant/datasets/ecp/ECP
SAVE_FILE = sys.argv[2]  # /home/vobecant/datasets/ecp/ECP
TRN_ANN_DIR = os.path.join(DATASET_DIR, 'day', 'labels', 'train')
VAL_ANN_DIR = os.path.join(DATASET_DIR, 'day', 'labels', 'val')

trn_files = [os.path.join(TRN_ANN_DIR, f) for f in Path(TRN_ANN_DIR).rglob('*.json')]
val_files = [os.path.join(VAL_ANN_DIR, f) for f in Path(VAL_ANN_DIR).rglob('*.json')]

d = os.path.split(SAVE_FILE)[0]
if not os.path.exists(d):
    os.makedirs(d)

'''
Annotations for CSP training are in the form of list of dictionaries. Each such dict contains:
'ignoreareas'
'vis_bboxes'
'bboxes'
'filepath'
'''

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

    for ann in trn['children']:
        bbox = [ann['x0'], ann['y0'], ann['x1'], ann['y1']]
        if ann['identity'] == 'pedestrian':
            ann['bboxes'].append(bbox)
        else:
            ann['ignoreareas'].append(bbox)

    ann['bboxes'] = np.asarray(ann['bboxes'])
    ann['ignoreareas'] = np.asarray(ann['ignoreareas'])

    anns_list.append(ann)

with open(SAVE_FILE, 'rb') as f:
    pickle.dump(anns_list, f)
