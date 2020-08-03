import os
import sys
import cv2
from PIL import Image

DIR = sys.argv[1]

imgs = [im for im in os.listdir(DIR) if '_' not in im]

for im in imgs:
    full_fname = os.path.join(DIR, im)
    crop_fname, ext = im.split('.')
    crop_fname = crop_fname + '_0.png'
    crop_fname = os.path.join(DIR, crop_fname)
    full = Image.open(full_fname)
    crop = Image.open(crop_fname)
    full.show()
    crop.show()
