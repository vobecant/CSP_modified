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

CHOOSEN_IMAGES = [
    '58c58332bc2601370015abd4',
    '58c58332bc2601370015abd2',
    '58c58332bc2601370015abe5',
    '58c580adbc260137e0956f58',  # O
    '58c580b3bc260137e0957e91',  # O
]


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[0] = x[0]
    y[1] = x[1]
    y[2] = x[0] + x[2]  # bottom right x
    y[3] = x[1] + x[3]  # bottom right y
    return y


def save_crop(bb, image, save_file):
    side = int(max(bb[-2:]) * 1.5)
    x_c = bb[0] + bb[2] // 2
    y_c = bb[1] + bb[3] // 2

    left = max(0, x_c - side // 2)
    top = max(0, y_c - side // 2)
    right = min(2047, left + side)
    bottom = min(1023, top + side)

    crop = image[top:bottom, left:right]
    cv2.imwrite(save_file, crop)


def plot_bbs(image, image_name, bbs, vis, heights, save_dir, color):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # TODO: plot whole image
    for i, (bb, v, h) in enumerate(zip(bbs, vis, heights)):
        bb_xyxy = bb  # xywh2xyxy(bb)
        plot_one_box(bb_xyxy, image, color, 'v{:.2f}|h{}'.format(v, h))
        # TODO: save crop of the missed sample
        save_file_crop = os.path.join(save_dir, image_name + '_{}.png'.format(i))
        save_crop(bb, image, save_file_crop)
    save_file_scene = os.path.join(save_dir, image_name + '.jpg')
    cv2.imwrite(save_file_scene, image)


def plot_one_box(bb, img, color, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or 1  # line/font thickness
    c1, c2 = (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = c2[0] - t_size[0], c2[1] + t_size[1] + 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


gt_anns = sys.argv[1]  # original /home/vobecant/PhD/CSP/data/cache/nightowls/train_h50_nonempty_xyxy
# trn_img_dir = sys.argv[2] # /home/vobecant/datasets/nightowls/nightowls_training
save_dir = sys.argv[2]  # /home/vobecant/PhD/CSP/nightowls_analysis/training_set

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir_plots = os.path.join(save_dir, 'plots')
if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)

with open(gt_anns, 'r') as f:
    anns = json.load(f)

N_CHOOSEN = 500
CHOOSEN_IDS = np.random.randint(0, len(anns) - 1, N_CHOOSEN)

# color_ours = (31, 119, 180)
color_ours = (144, 238, 144)
color_paper = (255, 127, 14)
color_gt = (100, 149, 237)  # (44, 160, 44)
GT_TL = 1

heights = []
n_occluded = 0
for i, ann in enumerate(anns):
    bbs_gt = ann['bboxes']
    vis = [1 for _ in bbs_gt]
    hs = []
    image_path = ann['filepath']
    _, image_name = os.path.split(image_path)
    for bb in ann['bboxes']:
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        heights.append(h)
        hs.append(h)

    img_name_noext = image_name.split('.')[0]
    if img_name_noext not in CHOOSEN_IMAGES:
        print('Skip {}'.format(i))
        continue
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plot_bbs(image, img_name_noext, bbs_gt, vis, hs, save_dir, color_gt)

n_reasonable = len(heights) - n_occluded  # should be 26348
print('Number of samples: {}\nreasonable: {}\noccluded:{}'.format(len(heights), n_reasonable, n_occluded))

fig, axs = plt.subplots(1, 1, tight_layout=True)
n_bins = 40
axs.hist(heights, bins=n_bins)
axs.set_title('training heights')
plt.savefig(os.path.join(save_dir_plots, 'trn_hists_heights.jpg'))
plt.close()

fig, axs = plt.subplots(1, 1, tight_layout=True)
n_bins = 40
axs.hist(heights, bins=n_bins, cumulative=True, density=True)
axs.set_title('training heights')
plt.savefig(os.path.join(save_dir_plots, 'trn_hists_heights_cumulative.jpg'))
plt.close()

data = {
    'heights': heights,
    # 'visibilities': visibilities
}

with open('/home/vobecant/PhD/CSP/nightowls_analysis/train_statistics.pkl', 'wb') as f:
    pickle.dump(data, f)
