from __future__ import print_function
import os, sys
import json
import time

"""
Author: Antonin Vobecky
Mail: a.vobecky@gmail.com, vobecant@fel.cvut.cz
Github: https://github.com/vobecant
LinkedIn: https://www.linkedin.com/in/antoninvobecky/

Usage:
    Python reimplementation of MATLAB code in dt_txt2json.m.
    The script takes as the first argument a path to the folder with directories that contain detection results (corresponds to main_path).
    If the path is not given, use default '../output/valresults/city/h/off'.
    The script processes each detections file and saves its JSON version to the same folder.
"""


def txt2jsonFile(res, test_imgs):
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


def convert_file(dt_path, test_imgs):
    with open(dt_path, 'r') as f:
        res = f.readlines()
    out_path = dt_path.replace('.txt', '.json')
    res_json = txt2jsonFile(res, test_imgs)
    with open(out_path, 'w') as f:
        json.dump(res_json, f)
    return out_path


def find_txt_det_file(d):
    files = os.listdir(d)
    for tmp in files:
        if 'det' in tmp and 'txt' in tmp:
            return os.path.join(d, tmp)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_path = sys.argv[1]
        print('Look for models in "{}"'.format(main_path))
    else:
        main_path = '../output/valresults/city/h/off'
        print('Use default path "{}"'.format(main_path))
    if len(sys.argv) > 2:
        test_imgs = int(sys.argv[2])
        print('There are {} test images.'.format(test_imgs))
    else:
        test_imgs = None

    start_t = time.time()

    dirs = [os.path.join(main_path, d) for d in os.listdir(main_path)]
    print('Found {} directories with detections.\n'.format(len(dirs)))
    dirs = sorted(dirs)

    for d in dirs:
        if 'best' in d:
            print('Skipping {}'.format(d))
            continue
        ndt = 0
        dt_coco = {}
        dt_path = find_txt_det_file(d)
        if dt_path is None:
            continue
        print('Processing detections from file {}'.format(dt_path))
        out_path = convert_file(dt_path, test_imgs=test_imgs)
        print('Saved detections to {}\n'.format(out_path))

    elapsed_t = time.time() - start_t
    print('Conversion completed! Total time {:.1f}s'.format(elapsed_t))
