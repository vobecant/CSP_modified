from __future__ import print_function
import os, sys

import numpy as np
import shutil
from coco import COCO
from eval_MR_multisetup import COCOeval
import cv2
import matplotlib.pyplot as plt


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
        bb_xyxy = xywh2xyxy(bb)
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


detections_file = sys.argv[1] # /home/vobecant//PhD/CSP/output/valresults/nightowls/h/off_orig_lr0.0008_baseline/117/val_det.txt
print('Given file {} with detections'.format(detections_file))
res_file = None
resFile_txt = os.path.join(detections_file)
resFile = os.path.join(detections_file.replace('.txt', '.json'))
annFile = '/home/vobecant/datasets/nightowls/nightowls_validation.json'
base_save_dir = './missed_detections'
img_base = '/home/vobecant/datasets/nightowls/nightowls_validation'

for id_setup in range(4):
    cocoGt = COCO(annFile)
    img_lut = {img['id']: img for img in cocoGt.imgs.values()}
    ann_lut = {ann['id']: ann for ann in cocoGt.anns.values()}
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    setup_name = cocoEval.params.SetupLbl[id_setup]
    setup_savedir = os.path.join(base_save_dir, setup_name)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate(id_setup)
    misses = cocoEval.accumulate(plot=True, return_misses=True)
    mean_mr = cocoEval.summarize(id_setup, res_file)
    missed_heights = []
    missed_visibilities = []
    for img_id, ms in misses.items():
        if len(ms):
            image_name = img_lut[img_id]['file_name']
            image_path = os.path.join(img_base, image_name)
            image = cv2.imread(image_path)
            bbs = [ann_lut[m]['bbox'] for m in ms]
            vis = [ann_lut[m]['vis_ratio'] for m in ms]
            missed_visibilities.extend(vis)
            heights = [bb[-1] for bb in bbs]
            missed_heights.extend(heights)
            plot_bbs(image, image_name.split('.')[0], bbs, vis, heights, setup_savedir,
                     color=(0, 0, 255))
    # TODO: plot 2D histogram
    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist2d(missed_heights, missed_visibilities)
    plt.title('Visibility and height of missed {} ({}).'.format(setup_name, len(missed_heights)))
    plt.colorbar(hist[3], ax=ax)
    plt.savefig(os.path.join(setup_savedir, 'test_heightVis_hist_missed_{}.jpg'.format(setup_name)))
    plt.close()
    if id_setup == 0:
        mr_reasonable = mean_mr
