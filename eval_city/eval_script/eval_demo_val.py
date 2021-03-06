from __future__ import print_function
import os, sys
from coco import COCO
from eval_MR_multisetup import COCOeval


def eval_json_reasonable(annFile,resFile):
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


annType = 'bbox'  # specify type here

# parse experiment name
assert len(sys.argv)==2, "You must specify path to the root directory with results!"
main_path = sys.argv[1]
print("Given experiment name: '{}'.".format(sys.argv[1]))

# initialize COCO ground truth api
annFile = '/home/vobecant/PhD/CSP/eval_city/val_gt.json'
print('Looking for detections in {}'.format(main_path))

for f in sorted(os.listdir(main_path)):
    print('file: {}'.format(f))
    # initialize COCO detections api
    dt_path = os.path.join(main_path, f)
    resFile = os.path.join(dt_path, 'val_dt.json')
    respath = os.path.join(dt_path, 'results.txt')
    # if os.path.exists(respath):
    #     continue
    ## running evaluation
    if not os.path.exists(resFile):
        print("Skipping {} ... Doesn't exist yet.".format(resFile))
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
        cocoEval.summarize(id_setup, res_file)
    print('')

    res_file.close()
