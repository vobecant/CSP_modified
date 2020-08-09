import json
import pickle
import os
import sys
from pathlib import Path

DATASET_DIR = sys.argv[1] # /home/vobecant/datasets/ecp/ECP
TRN_IMG_DIR = os.path.join(DATASET_DIR, 'day', 'img', 'train')
VAL_IMG_DIR = os.path.join(DATASET_DIR, 'day', 'img', 'val')
TRN_ANN_DIR = os.path.join(DATASET_DIR, 'day', 'labels', 'train')
VAL_ANN_DIR = os.path.join(DATASET_DIR, 'day', 'labels', 'val')

trn_files = [os.path.join(TRN_ANN_DIR, f) for f in Path(TRN_ANN_DIR).rglob('*.json')]
val_files = [os.path.join(VAL_ANN_DIR, f) for f in Path(VAL_ANN_DIR).rglob('*.json')]

'''
Annotations for CSP training are in the form of list of dictionaries. Each such dict contains:
'ignoreareas'
'vis_bboxes'
'bboxes'
'filepath'
'''

trn_anns = []
for trnf in trn_files:
    with open(trnf, 'r') as f:
        trn = json.load(f)

    img_path = trnf.replace('.json','.png').replace('/labels/','/img/')
    assert os.path.exists(img_path)

    ann = {'ignoreareas': [],
           'vis_bboxes': [],
           'bboxes': [],
           'filepath': trn}

