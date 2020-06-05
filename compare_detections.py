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

CHOOSEN_IDS = [234, 475, 41, 46, 69, 75, 84, 131, 135, 144, 145, 147, 152, 155, 162, 186, 189, 194, 204, 231, 253, 292, 328, 351, 375, 412, 471, 481]  # [234, 475]
CONF_THRES = 0.3

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


def plot_images(img, boxes, confs, path=None, fname='images.jpg', gt=False, label=None, color=(255, 255, 255),
                tlg=None):
    boxes = np.asarray(boxes).reshape((-1, 4))
    boxes[:, 2:] += boxes[:, :2]
    tl = tlg if tlg is not None else 2  # line thickness
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
            if gt or confs[j] > CONF_THRES:  # 0.3 conf thresh
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
dt_file2 = sys.argv[2]
val_img_dir = sys.argv[3]  # /home/vobecant/datasets/cityscapes/leftImg8bit/val
save_dir = sys.argv[4]
dt_gt_file = 'eval_city/val_gt.json'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(dt_file1, 'r') as f:
    dets1 = json.load(f)

with open(dt_file2, 'r') as f:
    dets2 = json.load(f)

with open(dt_gt_file, 'r') as f:
    gts = json.load(f)

dets1_byImg = {i: {'boxes': [], 'scores': []} for i in range(1, 501)}
dets2_byImg = {i: {'boxes': [], 'scores': []} for i in range(1, 501)}
bbs_gt_all = {i: [] for i in range(1, 501)}

# color_ours = (31, 119, 180)
color_ours = (144, 238, 144)
color_paper = (255, 127, 14)
color_gt = (100, 149, 237)  # (44, 160, 44)
GT_TL = 1

for dt in dets1:
    dets1_byImg[dt['image_id']]['boxes'].append(dt['bbox'])
    dets1_byImg[dt['image_id']]['scores'].append(dt['score'])

for dt in dets2:
    dets2_byImg[dt['image_id']]['boxes'].append(dt['bbox'])
    dets2_byImg[dt['image_id']]['scores'].append(dt['score'])

for ann in gts['annotations']:
    if ann['category_id'] != 1 or ann['ignore'] or ann['iscrowd']:  # or ann['vis_ratio'] < 0.65 or ann['height'] < 50
        continue
    image_id = ann['image_id']
    bbs_gt_all[image_id].append(ann['bbox'])

for i, (dt1, dt2) in enumerate(zip(dets1_byImg.values(), dets2_byImg.values())):
    if len(CHOOSEN_IDS) and (i + 1) not in CHOOSEN_IDS:
        continue

    bbs_gt = bbs_gt_all[i + 1]

    image_name = gts['images'][i]['im_name']
    print('{} {}'.format(i + 1, image_name))
    city = image_name.split('_')[0]
    image_name = os.path.join(val_img_dir, city, image_name)
    image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

    bbs1, scores1 = dt1['boxes'], dt1['scores']

    for tl in [1, 2]:
        img_dts_ours = plot_images(image.copy(), bbs1, scores1, None, label=None, color=color_ours, tlg=tl)
        plt.imsave(os.path.join(save_dir, 'im{}_dets_ours_tl{}.png'.format(i + 1, tl)), img_dts_ours)

    continue

    img_dts_ours_gt = plot_images(img_dts_ours.copy(), bbs_gt, None, None, label=None, color=color_gt, tlg=GT_TL,
                                  gt=True)
    img_dts_all = plot_images(image.copy(), bbs1, scores1, image_name, label='ours', color=color_ours, tlg=2)
    plt.imsave(os.path.join(save_dir, 'im{}_dets_ours.png'.format(i + 1)), img_dts_ours)
    plt.imsave(os.path.join(save_dir, 'im{}_dets_ours_wGT_tl{}.png'.format(i + 1, GT_TL)), img_dts_ours_gt)

    bbs2, scores2 = dt2['boxes'], dt2['scores']
    img_dts_paper = plot_images(image.copy(), bbs2, scores2, None, label=None, color=color_paper)
    img_dts_paper_gt = plot_images(img_dts_paper.copy(), bbs_gt, None, None, label=None, color=color_gt, tlg=GT_TL,
                                   gt=True)
    plt.imsave(os.path.join(save_dir, 'im{}_dets_paper.png'.format(i + 1)), img_dts_paper)
    plt.imsave(os.path.join(save_dir, 'im{}_dets_paper_wGT_tl{}.png'.format(i + 1, GT_TL)), img_dts_paper_gt)
    img_dts_all = plot_images(img_dts_all, bbs2, scores2, image_name, label='paper', color=color_paper, tlg=2)

    plt.imsave(os.path.join(save_dir, 'im{}_dets_ours_paper.png'.format(i + 1)), img_dts_all)

    img_dts_all = plot_images(img_dts_all, bbs_gt, None, image_name, label='GT', gt=True, color=color_gt, tlg=1)

    plt.imsave(os.path.join(save_dir, 'im{}_dets.png'.format(i + 1)), img_dts_all)
