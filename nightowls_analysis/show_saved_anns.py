import os
import sys
import cv2

DIR = sys.argv[1]

imgs = [im for im in os.listdir(DIR) if '_' not in im]

for im in imgs:
    full_fname = os.path.join(DIR, im)
    crop_fname, ext = im.split('.')
    crop_fname = crop_fname + '_0.png'
    crop_fname = os.path.join(DIR, crop_fname)
    full = cv2.imread(full_fname)
    crop = cv2.imread(crop_fname)
    cv2.imshow('crop', crop)
    cv2.imshow('full', full)
    cv2.waitKey()
