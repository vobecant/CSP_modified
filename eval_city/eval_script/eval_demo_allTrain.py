from __future__ import print_function
import os, sys

import numpy as np
import shutil
from coco import COCO
from eval_MR_multisetup import COCOeval
import cv2


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
if os.path.isfile(main_path):
    print('Given file {} with detections'.format(main_path))
    res_file = None
    resFile_txt = os.path.join(main_path)
    resFile = os.path.join(main_path.replace('.txt', '.json'))
    for id_setup in range(6):
        cocoGt = COCO(annFile)
        setup_name = ''
        img_lut = {img['id']: img for img in cocoGt.imgs}
        ann_lut = {ann['id']: ann for ann in cocoGt.anns}
        cocoDt = cocoGt.loadRes(resFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        misses = cocoEval.accumulate(plot=True, return_misses=True)
        mean_mr = cocoEval.summarize(id_setup, res_file)
        for img_id, ms in misses:
            if len(ms):
                image_name = img_lut[img_id]
                city = image_name.split('_')[0]
                image_path = os.path.join(img_base, city, image_name)
                image = cv2.imread(image_path)
                bbs = [ann_lut[m]['bbox'] for m in ms]
                vis = [ann_lut[m]['vis_ratio'] for m in ms]
                plot_bbs(image,bbs,vis,os.path.join(base_save_dir,setup_name))
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
