from __future__ import print_function
import os, sys
from coco import COCO
from eval_MR_multisetup import COCOeval

annType = 'bbox'  # specify type here
ious = [0.5, 0.7]

# parse experiment name
if len(sys.argv) == 1:
    exp_name = ''
    print('No experiment name given. Run with default parameters.')
else:
    exp_name = '_{}'.format(sys.argv[1])
    print("Given experiment name: '{}'.".format(sys.argv[1]))

# initialize COCO ground truth api
annFile = '../../data/precarious_dataset/test_annotations.json'
main_path = '../../output/valresults/precarious/h/off{}'.format(exp_name)

cocoGt = COCO(annFile)
resFile = 'C:\\Users\\Antonin-PC\\school\\PhD\\projects\\CSP_modified\\val_dt.json'  # os.path.join(dt_path, 'val_dt.json')
cocoDt = cocoGt.loadRes(resFile)
imgIds = sorted(cocoGt.getImgIds())

for f in sorted(os.listdir(main_path)):
    print('file: {}'.format(f))
    # initialize COCO detections api
    dt_path = os.path.join(main_path, f)
    resFile = os.path.join(dt_path, 'val_dt.json')
    print('Load results from {}'.format(resFile))
    respath = os.path.join(dt_path, 'results.txt')
    # if os.path.exists(respath):
    #     continue
    ## running evaluation
    if not os.path.exists(resFile):
        print("Skipping {} ... Doesn't exist yet.".format(resFile))
        continue
    res_file = open(respath, "w")
    for iou_thr in ious:
        for id_setup in range(1):
            print('Load annFile {}'.format(annFile))
            cocoGt = COCO(annFile)
            print('Load resFile {}'.format(resFile))
            cocoDt = cocoGt.loadRes(resFile)
            imgIds = sorted(cocoGt.getImgIds())
            cocoEval = COCOeval(cocoGt, cocoDt, annType)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate(id_setup)
            cocoEval.accumulate()
            cocoEval.summarize(id_setup, res_file, iou_thr=iou_thr)
    print('')

    res_file.close()
