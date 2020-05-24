from __future__ import print_function
import os, sys

import numpy as np
import shutil
from coco import COCO
from eval_MR_multisetup import COCOeval


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
main_path = sys.argv[1]
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
            mr_reasonable = float(f.readline())
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
