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

# parse experiment name
if len(sys.argv) == 1:
    exp_name = ''
    print('No experiment name given. Run with default parameters.')
else:
    exp_name = '_{}'.format(sys.argv[1])
    print("Given experiment name: '{}'.".format(sys.argv[1]))

# initialize COCO ground truth api
annFile = '/home/vobecant/PhD/CSP/data/cache/cityperson_trainValTest/val_gt_fullRes.json'
main_path = '../../output/valresults/city_valMR_eccv/h/off{}'.format(exp_name)
print('Looking for detections in {}'.format(main_path))

best_mr_reasonable = np.inf
best_mr_name = None

for f in sorted(os.listdir(main_path)):
    print('file: {}'.format(f))
    if 'val' in f:
        continue
    # initialize COCO detections api
    dt_path = os.path.join(main_path, f)
    resFile = os.path.join(dt_path, 'val_dt.json')
    respath = os.path.join(dt_path, 'results.txt')
    # if os.path.exists(respath):
    #     continue
    ## running evaluation
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
                print('New best validation MR with model {} : {} -> {}'.format(f, best_mr_reasonable, mr_reasonable))
                best_mr_reasonable = mr_reasonable
                best_mr_name = f
    print('')

    res_file.close()

print('best_mr_name: {}'.format(best_mr_name))
weights_path_orig = '../../output/valmodels/city_valMR/h/off{}'.format(exp_name)
weights_path_new = '../../output/valmodels/city_valMR_eccv/h/off{}'.format(exp_name)
if not os.path.exists(weights_path_new):
    os.makedirs(weights_path_new)
model_name = find_model(weights_path_orig, int(best_mr_name))
print('Best overall MR with model {} : {}'.format(model_name, best_mr_reasonable))
best_model_path = os.path.join(weights_path_orig, model_name)
tgt_path = os.path.join(weights_path_new, 'best_val.hdf5')
shutil.copy(best_model_path, tgt_path)
