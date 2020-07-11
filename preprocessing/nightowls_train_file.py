import sys
import os
import json
import pickle

TRAIN_ANNS = '/home/vobecant/datasets/nightowls/nightwols_training.json'
SAVE_FILE = '/home/vobecant/PhD/CSP/data/cache/nightowls/train_h50_allAnns_xyxy_baseline'
MIN_HEIGHT = 50
LABELS = [0]

with open(TRAIN_ANNS, 'r') as f:
    train_anns = json.load(f)

choosen_anns = []

with open(SAVE_FILE, 'wb') as f:
    pickle.dump(choosen_anns, f, protocol=2)
