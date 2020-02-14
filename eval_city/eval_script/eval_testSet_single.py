from __future__ import print_function
import os, sys
from coco import COCO
from eval_MR_multisetup import COCOeval

annType = 'bbox'  # specify type here

# initialize COCO ground truth api
annFile = '../val_gt.json'

dt_path = sys.argv[1]
resFile = os.path.join(dt_path, 'val_dt.json')
respath = os.path.join(dt_path, 'results.txt')
# if os.path.exists(respath):
#     continue
## running evaluation
if not os.path.exists(resFile):
    print("Skipping {} ... Doesn't exist yet.".format(resFile))
else:
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
