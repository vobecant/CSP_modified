from __future__ import print_function
import os
from coco import COCO
from eval_MR_multisetup import COCOeval

annType = 'bbox'  # specify type here

# initialize COCO ground truth api
annFile = '../val_gt.json'
main_path = '../../output/valresults/city/h/off'
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
        print("Skipping {} ... Doesn't exist yet.")
        continue
    res_file = open(respath, "w")
    for id_setup in range(0, 4):
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
