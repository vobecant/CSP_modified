import json
import sys
import os
from coco import COCO
from eval_MR_multisetup import COCOeval
import numpy as np
import heapq

# Ground truth
# annFile = '/home/vobecant/datasets/nightowls/nightowls_validation.json'
annFile = '/home/vobecant/datasets/nightowls/nightowls_validation_reasonable_nonempty.json'
main_path = sys.argv[1]
print('Looking for detections in {}'.format(main_path))
best_mr_reasonable = np.inf
best_mr_name = None


def txt2json(txt_path, json_path):
    with open(txt_path, 'r') as f:
        res = f.readlines()
    res_json = txt2jsonFile(res)
    with open(json_path, 'w') as f:
        json.dump(res_json, f)
    return json_path


def txt2jsonFile(res):
    out_arr = []
    for det in res:
        det = det.rstrip("\n\r").split(' ')
        img_id = int(float(det[0]))
        bbox = [float(f) for f in det[1:5]]
        score = float(det[5])
        det_dict = {'image_id': img_id,
                    'category_id': 1,
                    'bbox': bbox,
                    'score': score}
        out_arr.append(det_dict)
    return out_arr


def find_txt_det_file(d):
    files = os.listdir(d)
    for tmp in files:
        if 'det' in tmp and 'txt' in tmp:
            return os.path.join(d, tmp)

best_ordered = []

for f in sorted(os.listdir(main_path)):
    print('file: {}'.format(f))
    # Detections
    dt_path = os.path.join(main_path, f)
    resFile_txt = find_txt_det_file(dt_path)
    if resFile_txt is None:
        print('Didn\'t find detections for ep {}.'.format(f))
    resFile = os.path.join(dt_path, 'val_dt.json')
    respath = os.path.join(dt_path, 'results.txt')

    # convert txt detections to json detections
    txt2json(resFile_txt, resFile)

    filesize = os.path.getsize(resFile_txt)
    if filesize == 0:
        print("The file is empty: {}. No detections. Skipping".format(resFile_txt))
        continue
    if os.path.exists(respath):
        with open(respath) as fl:
            configs = ['Reasonable', 'Reasonable_small', 'heavy', 'All']
            for ci, config in enumerate(configs):
                line = fl.readline()
                perc = line.split(' ')[-1][:-2]
                res = float(perc)
                if ci == 0:
                    mr_reasonable = res / 100
                    heapq.heappush(best_ordered, (mr_reasonable, f))
                print('\t{}: {:.4f}%'.format(config, res))
        if mr_reasonable < best_mr_reasonable:
            print('New best test MR with model {} : {} -> {}'.format(f, best_mr_reasonable, mr_reasonable))
            best_mr_reasonable = mr_reasonable
            best_mr_name = f
        continue

    ## running evaluation
    res_file = open(respath, "w")
    for id_setup in range(0, 4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(resFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        mean_mr = cocoEval.summarize(id_setup, res_file)
        if id_setup == 0:
            # reasonable setup
            mr_reasonable = mean_mr
            heapq.heappush(best_ordered, (mr_reasonable, f))
            if mr_reasonable < best_mr_reasonable:
                print('New best test MR with model {} : {} -> {}'.format(f, best_mr_reasonable, mr_reasonable))
                best_mr_reasonable = mr_reasonable
                best_mr_name = f
    print('')
    res_file.close()

print('best_mr_name: {}'.format(best_mr_name))
print('Best overall MR with model {} : {}'.format(best_mr_name, best_mr_reasonable))
print('10 best models:')
for i in range(10):
    mr, name = heapq.heappop(best_ordered)
    print('{}: {}'.format(name, mr))
