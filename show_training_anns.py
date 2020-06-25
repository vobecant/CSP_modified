import json
import sys
import os
import cv2
import numpy as np
import math
import matplotlib
import random
import pickle

matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHOOSEN_IDS = [i for i in range(1000, 1000)]


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
    # boxes[:, 2:] += boxes[:, :2]
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
            plot_one_box(box, img, label=label, color=color, line_thickness=tl, gt=gt, paper=label == 'paper')

    # Draw image filename labels
    if path is not None:
        label = os.path.basename(path)[:40]  # trim to 40 char
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        cv2.putText(img, label, (0 + 5, 0 + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                    lineType=cv2.LINE_AA)

    if fname is not None:
        cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return img


trn_anns = sys.argv[1]  # original /home/vobecant/PhD/CSP/data/cache/cityperson/train_h50
# /home/vobecant/datasets/DummyGAN_cityscapes_occluded_final/1P # /home/vobecant/PhD/CSP/data/cityperson/images/train
trn_img_dir = sys.argv[2]
save_dir = sys.argv[3]  # /home/vobecant/datasets/DummyGAN_cityscapes_occluded_final/1P/gts # ./output/training_gts

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir_plots = os.path.join(save_dir, 'plots')
if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)

with open(trn_anns, 'rb') as f:
    anns = pickle.load(f, encoding='latin1')

# color_ours = (31, 119, 180)
color_ours = (144, 238, 144)
color_paper = (255, 127, 14)
color_gt = (100, 149, 237)  # (44, 160, 44)
GT_TL = 1

heights = []
visibilities = []
for i, ann in enumerate(anns):
    bbs_gt = ann['bboxes']
    image_name = ann['filepath']
    for bb, vbb in zip(ann['bboxes'], ann['vis_bboxes']):
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        heights.append(h)
        wv = vbb[2] - vbb[0]
        hv = vbb[3] - vbb[1]
        visibility = (wv * hv) / (w * h)
        visibilities.append(visibility)

    if len(CHOOSEN_IDS) and i not in CHOOSEN_IDS:
        print('Skip {}'.format(i))
        continue
    image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
    img_gts = plot_images(image, bbs_gt, None, image_name, label='GT', gt=True, color=color_gt, tlg=1)
    pth = os.path.join(save_dir, 'im{}_dets.png'.format(i + 1))
    print('Saved GTs to {}'.format(pth))
    plt.imsave(pth, img_gts)

fig, axs = plt.subplots(1, 2, tight_layout=True)
n_bins = 20
axs[0].hist(heights, bins=n_bins)
axs[0].set_title('training heights')
axs[1].hist(visibilities, bins=n_bins)
axs[1].set_title('training visibilities')
plt.savefig(os.path.join(save_dir_plots, 'height_hist_missed.jpg'))
plt.close()

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(heights, visibilities,
                 bins=[np.arange(50, max(heights), 10), np.arange(0, 1.0, 0.01)])
plt.title('Visibility and height of training samples.')
plt.colorbar(hist[3], ax=ax)
plt.savefig(os.path.join(save_dir_plots, 'heightVis_hist_training.jpg'))
plt.close()

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(heights, visibilities,
                 bins=[np.arange(50, 300, 10), np.arange(0, 1.0, 0.04)])
plt.title('Visibility and height of training samples, heights [50,300].')
plt.colorbar(hist[3], ax=ax)
plt.savefig(os.path.join(save_dir_plots, 'heightVis_hist_training_cutout.jpg'))
plt.close()

fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(heights, visibilities, density=True,
                 bins=[np.arange(50, 300, 10), np.arange(0, 1.0, 0.04)])
plt.title('Visibility and height of training samples, heights [50,300].')
plt.colorbar(hist[3], ax=ax)
plt.savefig(os.path.join(save_dir_plots, 'heightVis_hist_training_cutout_norm.jpg'))
plt.close()
