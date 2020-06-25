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
            if gt or confs[j] > 0.0:  # 0.3 conf thresh
                if gt:
                    plot_one_box(box, img, label=label + ' v{:.3f}'.format(confs[j]), color=color, line_thickness=tl,
                                 gt=gt, paper=label == 'paper')
                else:
                    plot_one_box(box, img, label=label + ' c{:.3f}'.format(confs[j]), color=color, line_thickness=tl,
                                 gt=gt, paper=label == 'paper')

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
save_dir_missed = os.path.join(save_dir, 'missed')
save_dir_plots = os.path.join(save_dir, 'plots')
dt_gt_file = sys.argv[4]  # /home/vobecant/PhD/CSP/data/cache/cityperson_trainValTest/train_h50_eccv_1P_hard_allTrain

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(save_dir_missed):
    os.makedirs(save_dir_missed)
if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)

with open(dt_file1, 'r') as f:
    dets1 = json.load(f)

with open(dt_gt_file, 'r') as f:
    gts = json.load(f)

dets1_byImg = {i: {'boxes': [], 'scores': []} for i in range(1, 501)}
bbs_gt_all = {i: [[], []] for i in range(1, 501)}
bbs_gt_all_ignore = {i: [] for i in range(1, 501)}

color_detected_reasonable = (0, 100, 0)
color_detected_occluded = (0, 128, 0)
color_missed_reasonable = (255, 0, 0)
color_missed_occluded = (255, 69, 0)
color_fp = (255, 140, 0)

for dt in dets1:
    dets1_byImg[dt['image_id']]['boxes'].append(dt['bbox'])
    dets1_byImg[dt['image_id']]['scores'].append(dt['score'])

n_peds_reasonable, n_peds_occluded = 0, 0
height_reasonable, height_occluded = [], []
vis_reasonable, vis_occluded = [], []
for ann in gts['annotations']:
    image_id = ann['image_id']
    height = ann['height']
    if ann['category_id'] != 1 or ann['ignore'] or ann['iscrowd'] or height < 50:
        bbs_gt_all_ignore[image_id].append(ann['bbox'])
        continue
    bbox = ann['bbox']
    vis_ratio = ann['vis_ratio']
    reasonable = vis_ratio >= 0.65

    if reasonable:
        bbs_gt_all[image_id][0].append((bbox, vis_ratio))
        n_peds_reasonable += 1
        vis_reasonable.append(vis_ratio)
        height_reasonable.append(height)
    else:
        bbs_gt_all[image_id][1].append((bbox, vis_ratio))
        n_peds_occluded += 1
        vis_occluded.append(vis_ratio)
        height_occluded.append(height)

# TODO: plot the distribution of occlusion levels in the reasonable and occluded subsets
print('Number of pedestrians > 50px:\n\treasonable: {}\n\toccluded: {}'.format(n_peds_reasonable, n_peds_occluded))
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
n_bins = 20
axs[0].hist(vis_reasonable, bins=n_bins)
axs[0].set_title('reasonable')
axs[1].hist(vis_occluded, bins=n_bins)
axs[1].set_title('occluded')
fig.suptitle('Visibility ratio')
plt.savefig(os.path.join(save_dir_plots, 'visibility_hist.jpg'))
plt.close()

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
n_bins = 20
axs[0].hist(height_reasonable, bins=n_bins)
axs[0].set_title('reasonable')
axs[1].hist(height_occluded, bins=n_bins)
axs[1].set_title('occluded')
fig.suptitle('Visibility ratio')
plt.savefig(os.path.join(save_dir_plots, 'height_hist.jpg'))
plt.close()

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(height_reasonable, vis_reasonable,
                 bins=[np.arange(50, max(height_reasonable), 10), np.arange(0.65, 1.0, 0.05)])
plt.title('Visibility and height of all reasonable.')
fig.colorbar(hist)
plt.savefig(os.path.join(save_dir_plots, 'heightVis_hist_all_reasonable.jpg'))
plt.close()

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(height_occluded, vis_occluded,
                 bins=[np.arange(50, max(height_occluded), 10), np.arange(0, 0.65, 0.05)])
plt.title('Visibility and height of all occluded.')
fig.colorbar(hist)
plt.savefig(os.path.join(save_dir_plots, 'heightVis_hist_all_occluded.jpg'))
plt.close()

image_paths = {im['id']: im['im_name'] for im in gts['images']}

start = time.time()


def overlap(xywh1, xywh2):
    '''

    :param xywh1: left, top, w, h
    :param xywh2:
    :return:
    '''
    xbox1 = [xywh1[0], xywh1[0] + xywh1[2]]
    ybox1 = [xywh1[1], xywh1[1] + xywh1[3]]
    xbox2 = [xywh2[0], xywh2[0] + xywh2[2]]
    ybox2 = [xywh2[1], xywh2[1] + xywh2[3]]

    overlaps = intersect1d(xbox1, xbox2) and intersect1d(ybox1, ybox2)
    return overlaps


def intersect1d(box1, box2):
    return box1[1] >= box2[0] and box2[1] >= box1[0]


def get_missed(detections, gts, iou_thr=0.5):
    missed = []
    heights = []
    visibilities = []
    for gt in gts:
        if gt[0][-1] < 50:
            continue
        detected = False
        for dt in detections:
            if overlap(dt, gt[0]) >= iou_thr:
                detected = True
                break
        if not detected:
            heights.append(gt[0][-1])
            visibilities.append(gt[1])
            missed.append(gt[0])
    return missed, heights, visibilities


missed_reasonable_height, missed_reasonable_visibility = [], []
missed_occluded_height, missed_occluded_visibility = [], []

for im_num, dt1 in enumerate(dets1_byImg.values()):
    image_name = image_paths[im_num + 1]
    city = image_name.split('_')[0]
    image_path = os.path.join(img_dir, city, image_name)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    bbs1, scores1 = dt1['boxes'], dt1['scores']
    bbs_gt_reasonable, bbs_gt_occluded = bbs_gt_all[im_num + 1]
    bbs_ignore = bbs_gt_all_ignore[im_num + 1]
    bbs_gt_both = bbs_gt_reasonable + bbs_gt_occluded

    # TODO: missed/detected reasonable
    detected_reasonable, detected_reasonable_scores = [], []
    missed_reasonable, h_r, v_r = [], [], []
    for gt in bbs_gt_reasonable:
        detected = False
        for i, (dt, conf) in enumerate(zip(bbs1, scores1)):
            if overlap(dt, gt[0]) >= 0.5:
                detected = True
                detected_reasonable.append(dt)
                del bbs1[i]
                detected_reasonable_scores.append(conf)
                del scores1[i]
                break
        if not detected:
            if gt[0][-1] < 50:
                # if it was not detected but is too small, do not report
                continue
            h_r.append(gt[0][-1])
            v_r.append(gt[1])
            missed_reasonable.append(gt[0])
    missed_reasonable_height.extend(h_r)
    missed_reasonable_visibility.extend(v_r)

    # TODO: missed/detected occluded
    detected_occluded, detected_occluded_scores = [], []
    missed_occluded, h_o, v_o = [], [], []
    for gt in bbs_gt_occluded:
        detected = False
        for i, (dt, conf) in enumerate(zip(bbs1, scores1)):
            if overlap(dt, gt[0]) >= 0.5:
                detected = True
                detected_occluded.append(dt)
                del bbs1[i]
                detected_occluded_scores.append(conf)
                del scores1[i]
                break
        if not detected:
            if gt[0][-1] < 50:
                # if it was not detected but is too small, do not report
                continue
            h_o.append(gt[0][-1])
            v_o.append(gt[1])
            missed_occluded.append(gt[0])
    missed_occluded_height.extend(h_o)
    missed_occluded_visibility.extend(v_o)

    # TODO: get false positives; do it by deleting the remaining detections and scores that are in ignore areas or contain non-pedestrian instances

    idx = 0
    while idx < len(bbs1):
        bb = bbs1[idx]
        ignored = False
        for ig in bbs_ignore:
            if overlap(bb, ig):
                ignored = True
                del bbs1[idx]
                del scores1[idx]
                break
        if not ignored:
            idx += 1

    image = image.copy()

    # TODO: plot correct detections
    image = plot_images(image, detected_reasonable, detected_reasonable_scores, image_name, label='det R', gt=False,
                        color=color_detected_reasonable)
    image = plot_images(image, detected_occluded, detected_occluded_scores, image_name, label='det O', gt=False,
                        color=color_detected_occluded)

    # TODO: plot false positives
    image = plot_images(image, bbs1, scores1, None, label='FP', gt=False, color=color_fp)

    # TODO: plot missed detections
    if len(missed_reasonable) or len(missed_occluded):
        if len(missed_reasonable):
            '''
            print('In {} missed reasonable:'.format(image_name))
            for h, o in zip(h_r, v_r):
                print('h={}, vis={:.3f}'.format(h, o))
            '''
            image = plot_images(image, missed_reasonable, v_r, image_name, label='miss R', gt=True,
                                color=color_missed_reasonable)
        if len(missed_occluded):
            '''
            print('In {} missed occluded:'.format(image_name))
            for h, o in zip(h_o, v_o):
                print('h={}, vis={:.3f}'.format(h, o))
            '''
            image = plot_images(image, missed_occluded, v_o, image_name, label='miss O', gt=True,
                                color=color_missed_occluded)
        plt.imsave(os.path.join(save_dir_missed, '{}_missed_dets.jpg'.format(image_name)), image)

    plt.imsave(os.path.join(save_dir, '{}_sorted_dets.jpg'.format(image_name)), image)
    if im_num % 50 == 0:
        print('{}/{} in {:.1f}s'.format(im_num, len(dets1_byImg), time.time() - start))

# TODO: plot statistics of the missed samples
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
# We can set the number of bins with the `bins` kwarg
n_bins = 20
axs[0].hist(missed_reasonable_visibility, bins=n_bins)
axs[0].set_title('reasonable ({})'.format(len(missed_reasonable_height)))
axs[1].hist(missed_occluded_visibility, bins=n_bins)
axs[1].set_title('occluded ({})'.format(len(missed_occluded_height)))
fig.suptitle('Visibility ratio, misses.')
plt.savefig(os.path.join(save_dir_plots, 'visibility_hist_missed.jpg'))
plt.close()

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
# We can set the number of bins with the `bins` kwarg
n_bins = 20
axs[0].hist(missed_reasonable_height, bins=n_bins)
axs[0].set_title('reasonable ({})'.format(len(missed_reasonable_height)))
axs[1].hist(missed_occluded_height, bins=n_bins)
axs[1].set_title('occluded ({})'.format(len(missed_occluded_height)))
fig.suptitle('Heights of misses.')
plt.savefig(os.path.join(save_dir_plots, 'height_hist_missed.jpg'))
plt.close()

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(missed_reasonable_height, missed_reasonable_visibility)
plt.title('Visibility and height of missed reasonable.')
plt.savefig(os.path.join(save_dir_plots, 'heightVis_hist_missed_reasonable.jpg'))
plt.close()

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(missed_occluded_height, missed_occluded_visibility)
plt.title('Visibility and height of missed occluded.')
plt.savefig(os.path.join(save_dir_plots, 'heightVis_hist_missed_occluded.jpg'))
plt.close()
