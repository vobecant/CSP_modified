from __future__ import print_function
import os, sys

import numpy as np
import shutil
from coco import COCO
from eval_MR_multisetup import COCOeval
import cv2
import matplotlib.pyplot as plt


def eval_json_reasonable(annFile, resFile):
    dt_path = os.path.split(resFile)[0]
    respath = os.path.join(dt_path, 'results.txt')
    res_file = open(respath, "w")
    mr_reasonable = None
    for id_setup in range(6):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(resFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        mean_mr = cocoEval.summarize(id_setup, res_file)
        if id_setup == 0:
            mr_reasonable = mean_mr
    print('')
    res_file.close()
    return mr_reasonable


def find_model(directory, epoch):
    files = os.listdir(directory)
    target = 'net_e{}_'.format(epoch)
    for f in files:
        if target in f:
            return f


annType = 'bbox'  # specify type here

# initialize COCO ground truth api
annFile = '/home/vobecant/PhD/CSP/eval_city/val_gt.json'
img_base = '/home/vobecant/datasets/cityscapes/leftImg8bit/val'
base_save_dir = './missed_detections'
main_path = sys.argv[1]


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


if os.path.isfile(main_path):
    print('Given file {} with detections'.format(main_path))
    res_file = None
    resFile_txt = os.path.join(main_path)
    resFile = os.path.join(main_path.replace('.txt', '.json'))
    for id_setup in range(6):
        cocoGt = COCO(annFile)
        img_lut = {img['id']: img for img in cocoGt.imgs.values()}
        ann_lut = {ann['id']: ann for ann in cocoGt.anns.values()}
        cocoDt = cocoGt.loadRes(resFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, annType)
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
                image_name = img_lut[img_id]['im_name']
                city = image_name.split('_')[0]
                image_path = os.path.join(img_base, city, image_name)
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
        hist = ax.hist2d(missed_heights, missed_visibilities, density=True)
        plt.title('Visibility and height of missed {} ({}).'.format(setup_name, len(missed_heights)))
        plt.colorbar(hist[3], ax=ax)
        plt.savefig(os.path.join(setup_savedir, 'test_heightVis_hist_missed_{}.jpg'.format(setup_name)))
        plt.close()
        if id_setup == 0:
            mr_reasonable = mean_mr
    print('')

else:
    print('Looking for detections in {}'.format(main_path))

    best_mr_reasonable = np.inf
    best_mr_name = None

    for f in sorted(os.listdir(main_path)):
        print('file: {}'.format(f))
        if 'val' in f:
            continue
        # initialize COCO detections api
        dt_path = os.path.join(main_path, f)
        resFile_txt = os.path.join(dt_path, 'val_det.txt')
        resFile = os.path.join(dt_path, 'val_dt.json')
        if not os.path.isfile(resFile):
            resFile = os.path.join(dt_path, 'val_det.json')
        respath = os.path.join(dt_path, 'results.txt')
        # if os.path.exists(respath):
        #     continue
        ## running evaluation
        filesize = os.path.getsize(resFile_txt)
        if filesize == 0:
            print("The file is empty: {}. No detections. Skipping".format(resFile_txt))
            continue
        if os.path.exists(respath):
            with open(respath) as f:
                configs = ['Reasonable', 'Reasonable_small', 'bare', 'partial', 'heavy', 'All']
                for ci, config in enumerate(configs):
                    res = float(f.readline())
                    if ci == 0:
                        mr_reasonable = res / 100
                    print('\t{}: {:.4f}%'.format(config, res))
            if mr_reasonable < best_mr_reasonable:
                print('New best test MR with model {} : {} -> {}'.format(f, best_mr_reasonable, mr_reasonable))
                best_mr_reasonable = mr_reasonable
                best_mr_name = f
            continue
        res_file = open(respath, "w")
        for id_setup in range(6):
            cocoGt = COCO(annFile)
            cocoDt = cocoGt.loadRes(resFile)
            imgIds = sorted(cocoGt.getImgIds())
            cocoEval = COCOeval(cocoGt, cocoDt, annType)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate(id_setup)
            cocoEval.accumulate()
            mean_mr = cocoEval.summarize(id_setup, res_file)
            if id_setup == 0:
                mr_reasonable = mean_mr
                if mr_reasonable < best_mr_reasonable:
                    print('New best test MR with model {} : {} -> {}'.format(f, best_mr_reasonable, mr_reasonable))
                    best_mr_reasonable = mr_reasonable
                    best_mr_name = f
        print('')

        res_file.close()

    print('best_mr_name: {}'.format(best_mr_name))
    weights_path_orig = main_path.replace('valresults', 'valmodels')
    model_name = find_model(weights_path_orig, int(best_mr_name))
    print('Best overall MR with model {} : {}'.format(model_name, best_mr_reasonable))
    best_model_path = os.path.join(weights_path_orig, model_name)
    tgt_path = os.path.join(weights_path_orig, 'best_val.hdf5')
    shutil.copy(best_model_path, tgt_path)
