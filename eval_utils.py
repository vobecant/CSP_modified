import json
import os

from eval_city.eval_script.coco import COCO


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


def convert_file(dt_path):
    if not os.path.exists(dt_path):
        print('File was not found! Skipping...')
        return None
    with open(dt_path, 'r') as f:
        res = f.readlines()
    dt_dir = os.path.split(dt_path)[0]
    out_path = os.path.join(dt_dir, 'val_dt.json')
    res_json = txt2jsonFile(res)
    with open(out_path, 'w') as f:
        json.dump(res_json, f)
    return out_path

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
