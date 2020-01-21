from __future__ import print_function
import os, sys
import json
import time


def txt2jsonFile(res):
    out_arr = []
    for det in res:
        img_id = int(det[0])
        bbox = [float(f) for f in det[1:5]]
        score = float(det[5])
        det_dict = {'image_id': img_id,
                    'category_id': 1,
                    'bbox': bbox,
                    'score': score}
        out_arr.append(det_dict)
    return out_arr


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_path = sys.argv[1]
        print('Look for models in "{}"'.format(main_path))
    else:
        main_path = '../output/valresults/city/h/off'
        print('Use default path "{}"'.format(main_path))

    start_t = time.time()

    dirs = [os.path.join(main_path, d) for d in os.listdir(main_path) if os.path.isdir(d)]
    print('Found {} directories with detections.\n'.format(len(dirs)))

    for d in dirs:
        ndt = 0
        dt_coco = {}
        dt_path = os.path.join(d, 'val_det.txt')
        print('Processing detections from file {}'.format(dt_path))
        if not os.path.exists(dt_path):
            print('File was not found! Skipping...')
            continue
        with open(dt_path, 'r') as f:
            res = f.readlines()
        out_path = os.path.join(d, 'val_dt.json')
        if os.path.exists(out_path):
            print('File was already processed. Skipping...')
            continue
        res_json = txt2jsonFile(res)

        with open(out_path, 'w') as f:
            json.dump(res_json, f)
        print('Saved detections to {}\n'.format(out_path))

    elapsed_t = time.time() - start_t
    print('Conversion completed! Total time {:.1f}s'.format(elapsed_t))
