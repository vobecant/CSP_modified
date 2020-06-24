import json
import sys
import os
import time

import cv2
import numpy as np
import math
import matplotlib
import random

import pickle

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


def plot_one_box(x, img, color=None, label=None, line_thickness=None, gt=False, paper=False):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        if gt:
            c1 = c2[0] - t_size[0], c2[1] + t_size[1] + 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        elif paper:
            c1 = c2[0] - t_size[0], c2[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1]), 0, tl / 3, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)
        else:
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_images(img, boxes, confs, path=None, fname='images.jpg', gt=False, label='', color=(255, 255, 255)):
    boxes = np.asarray(boxes).reshape((-1, 4))
    boxes[:, 2:] += boxes[:, :2]
    tl = 1  # 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness

    # un-normalise
    if np.max(img[0]) <= 1:
        img *= 255

    h, w, _ = img.shape  # height, width, channels

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    if len(boxes) > 0:
        classes = [1]
        for j, box in enumerate(boxes):
            if gt or confs[j] > 0.3:  # 0.3 conf thresh
                plot_one_box(box, img, label=label, color=color, line_thickness=tl, gt=gt, paper=label == 'paper')

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
img_dir = sys.argv[2]  # /home/vobecant/datasets/DummyGAN_cityscapes_hard/1P/images
save_dir = sys.argv[3]
dt_gt_file = sys.argv[4]  # /home/vobecant/PhD/CSP/data/cache/cityperson_trainValTest/train_h50_eccv_1P_hard_allTrain

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(dt_file1, 'r') as f:
    dets1 = json.load(f)

with open(dt_gt_file, 'r') as f:
    gts = json.load(f)

dets1_byImg = {i: {'boxes': [], 'scores': []} for i in range(1, 501)}
bbs_gt_all = {i: [[], []] for i in range(1, 501)}

color_ours = (31, 119, 180)
color_gt_reasonable = (44, 160, 44)
color_gt_occluded = (255, 127, 14)

for dt in dets1:
    dets1_byImg[dt['image_id']]['boxes'].append(dt['bbox'])
    dets1_byImg[dt['image_id']]['scores'].append(dt['score'])

for ann in gts['annotations']:
    if ann['category_id'] != 1 or ann['ignore'] or ann['iscrowd']:  # or ann['vis_ratio'] < 0.65 or ann['height'] < 50
        continue
    image_id = ann['image_id']

    bbox = ann['bbox']
    reasonable = ann['vis_ratio'] >= 0.65

    if reasonable:
        bbs_gt_all[image_id][0].append(bbox)
    else:
        bbs_gt_all[image_id][1].append(bbox)

image_paths = {im['id']: im['im_name'] for im in gts['images']}

start = time.time()
for i, dt1 in enumerate(dets1_byImg.values()):
    image_name = image_paths[i+1]
    city = image_name.split('_')[0]
    image_path = os.path.join(img_dir,city,image_name)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    bbs1, scores1 = dt1['boxes'], dt1['scores']
    img_dt1 = plot_images(image.copy(), bbs1, scores1, image_name, label='ours', color=color_ours)

    bbs_gt_reasonable, bbs_gt_occluded = bbs_gt_all[i + 1]
    img_dts = plot_images(img_dt1, bbs_gt_reasonable, None, image_name, label='GT', gt=True, color=color_gt_reasonable)
    img_dts = plot_images(img_dts, bbs_gt_occluded, None, image_name, label='GT', gt=True, color=color_gt_occluded)

    plt.imshow(img_dts)
    plt.show()
    plt.imsave(os.path.join(save_dir, '{}_dets.jpg'.format(image_name)), img_dts)
    if i % 50 == 0:
        print('{}/{} in {:.1f}s'.format(i, len(dets1_byImg), time.time() - start))
