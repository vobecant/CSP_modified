import json
import sys
import os
import cv2
import numpy as np
import math
import matplotlib
import random

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_images(img, boxes, confs, path=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    if os.path.isfile(fname):  # do not overwrite
        return None

    # un-normalise
    if np.max(img[0]) <= 1:
        img *= 255

    h, w, _ = img.shape  # height, width, channels

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    if len(boxes) > 0:
        classes = [1]

        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        for j, box in enumerate(boxes.T):
            cls = int(classes[j])
            color = color_lut[cls % len(color_lut)]
            cls = names[cls] if names else cls
            if gt or confs[j] > 0.3:  # 0.3 conf thresh
                label = '%s' % cls if gt else '%s %.1f' % (cls, confs[j])
                plot_one_box(box, img, label=label, color=color, line_thickness=tl)

    # Draw image filename labels
    if path is not None:
        label = os.path.basename(path)[:40]  # trim to 40 char
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        cv2.putText(img, label, (0 + 5, 0 + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                    lineType=cv2.LINE_AA)

    # Image border
    # cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return img


dt_file1 = sys.argv[1]
dt_file2 = sys.argv[2]
dt_gt_file = 'eval_city/val_gt.json'

with open(dt_file1, 'r') as f:
    dets1 = json.load(f)

with open(dt_file2, 'r') as f:
    dets2 = json.load(f)

with open(dt_gt_file, 'r') as f:
    gt = json.load(f)

dets1_byImg = {i: {'boxes': [], 'scores': []} for i in range(500)}
dets2_byImg = {i: [] for i in range(500)}

for dt in dets1:
    dets1_byImg[dt['image_id']]['boxes'].append(dt['bbox'])
    dets1_byImg[dt['image_id']]['scores'].append(dt['score'])

for dt in dets2:
    dets2_byImg[dt['image_id']]['boxes'].append(dt['bbox'])
    dets2_byImg[dt['image_id']]['scores'].append(dt['score'])

for i, (dt1, dt2) in enumerate(zip(dets1_byImg.values(), dets2_byImg.values())):
    image_name = ''
    image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

    bbs1, scores1 = dt1['boxes'], dt1['scores']
    img_dt1 = plot_images(image.copy(), bbs1, scores1, image_name)

    bbs2, scores2 = dt2['boxes'], dt2['scores']
    img_dt2 = plot_images(image.copy(), bbs2, scores2, image_name)

    plt.imsave('det1.png', img_dt1)
    plt.imsave('det2.png', img_dt2)
