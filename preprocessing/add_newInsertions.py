import json
import pickle
import os
import sys
import numpy as np

# /home/vobecant/PhD/cut_paste_learn__data_generator/cpl_ins2cityscapes_final2_random/annotations
JSON_ANNS_DIR = sys.argv[1]
IMG_DIR = sys.argv[2]  # /home/vobecant/PhD/cut_paste_learn__data_generator/cpl_ins2cityscapes_final2_random/images
ORIG_ANNS = sys.argv[3]  # /home/vobecant/PhD/CSP/data/cache/cityperson/train_h50
SAVE_PATH = sys.argv[4] # /home/vobecant/PhD/CSP/data/cache/cityperson/train_h50_cpl

json_anns = [os.path.join(JSON_ANNS_DIR, a) for a in os.listdir(JSON_ANNS_DIR) if a.endswith('.json')]

with open(ORIG_ANNS, 'rb') as f:
    orig_anns = pickle.load(f, encoding='latin1')

img2orig = {os.path.split(ann['filepath'])[-1]: ann for ann in orig_anns}
img2id = {os.path.split(ann['filepath'])[-1]: i for i,ann in enumerate(orig_anns)}

for pth in json_anns:
    with open(pth, 'r') as f:
        ann = json.load(f)
    img_name = os.path.split(ann['bg_path'])[-1]
    orig_ann = img2orig[img_name]
    orig_ann['bboxes'] = np.concatenate((orig_ann['bboxes'], np.asarray(ann['xyxy']).reshape((-1,4))),axis=0)

with open(SAVE_PATH, 'wb') as f:
    pickle.dump(orig_anns,f)