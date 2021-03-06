import json
import sys, os
import numpy as np

if sys.version_info[0] == 2:
    import cPickle as pickle

    py = 2
else:
    import pickle

    py = 3

VAL_CITIES = ['aachen', 'bochum', 'dusseldorf']


def get_city(filename):
    city = os.path.split(filename)[1].split('.')[0].split('_')[0]
    return city


if __name__ == '__main__':
    orig_path = sys.argv[1]
    exp_name = '_'.join(os.path.split(orig_path)[1].split('_')[1:])
    if len(sys.argv) == 4:
        tst_height = int(sys.argv[2])
        tst_width = int(sys.argv[3])
        mult_h = tst_height / 1024.0
        mult_w = tst_width / 2048.0
        new_fname_val_json = 'val_gt.json'
    else:
        tst_height = 1024
        tst_width = 2048
        mult_h, mult_w = 1, 1
        new_fname_val_json = 'val_gt_fullRes.json'
    new_fname_train = 'train_{}'.format(exp_name)
    new_fname_val = 'val'

    # step 1
    # create new training split from the EXTENDED original training split
    if not os.path.exists(new_fname_train):
        with open(orig_path, 'rb') as f:
            if py == 3:
                orig_train = pickle.load(f, encoding='latin1')
            else:
                orig_train = pickle.load(f)

        train_anns = []
        for ann in orig_train:
            city = get_city(ann['filepath'])
            if city not in VAL_CITIES:
                train_anns.append(ann)

        print('New training set size: {} -> {} left for validation.'.format(len(train_anns),
                                                                            len(orig_train) - len(train_anns)))
        with open(new_fname_train, 'wb') as f:
            pickle.dump(train_anns, f, protocol=2)
    else:
        print('Skipping creating training cache file {} . Already exists.'.format(new_fname_train))

    # step 2
    # create new validation split from the NONEXTENDED original training split
    if not os.path.exists(new_fname_val) or not os.path.exists(new_fname_val_json):
        print('Create new JSON files.')
        val_gt_json = {'categories': [{"id": 1, "name": "pedestrian"}, {"id": 2, "name": "rider"},
                                      {"id": 3, "name": "sitting person"}, {"id": 4, "name": "other person"},
                                      {"id": 5, "name": "people group"}, {"id": 0, "name": "ignore region"}],
                       'images': [], 'annotations': []}

        with open('../cityperson/train_h50', 'rb') as f:
            if py == 3:
                orig_train = pickle.load(f, encoding='latin1')
            else:
                orig_train = pickle.load(f)

        val_anns = []
        images_added = {}
        images = []
        for ann in orig_train:
            city = get_city(ann['filepath'])
            if city in VAL_CITIES:
                val_anns.append(ann)
                image_name = os.path.split(ann['filepath'])[1]
                if image_name not in images_added.keys():
                    images_added[image_name] = len(images_added) + 1
                    val_gt_json['images'].append({'id': len(images_added),
                                                  'im_name': image_name,
                                                  'height': tst_height, 'width': tst_width})
                for bbox, vis_bbox in zip(ann['bboxes'], ann['vis_bboxes']):
                    x1, y1, x2, y2 = bbox
                    x1, x2 = x1 * mult_w, x2 * mult_w
                    y1, y2 = y1 * mult_h, y2 * mult_h
                    w, h = x2 - x1, y2 - y1
                    assert all([w > 0, h > 0]), 'w={},h={}, prev: {}'.format(w, h, bbox)
                    bbox = [int(b) for b in [x1, y1, w, h]]
                    x1v, y1v, x2v, y2v = vis_bbox
                    x1v, x2v = x1v * mult_w, x2v * mult_w
                    y1v, y2v = y1v * mult_h, y2v * mult_h
                    w_v, h_v = x2v - x1v, y2v - y1v
                    assert all([w_v > 0, h_v > 0]), 'w={},h={}, prev: {}'.format(w_v, h_v, vis_bbox)
                    vis_bbox = [int(vb) for vb in [x1v, y1v, w_v, h_v]]
                    assert all([x1 + w < tst_width, x1 >= 0, x1v + w_v < tst_width, x1v >= 0]), bbox
                    assert all([y1 + h < tst_height, y1 >= 0, y1v + h_v < tst_height, y1v >= 0]), bbox
                    vis_ratio = float((vis_bbox[-2] * vis_bbox[-1])) / float((bbox[-2] * bbox[-1]))
                    idx = len(val_gt_json['annotations']) + 1
                    height = int(bbox[-1])
                    print('Add ID: {}, image_id: {},  bbox: {}, vis_bbox: {}, height: {}, '
                          'vis_ratio: {}'.format(idx, images_added[image_name], bbox, vis_bbox, height, vis_ratio))
                    ann_json = {"id": idx, "image_id": images_added[image_name],
                                "category_id": 1, "iscrowd": 0, "ignore": 0,
                                "bbox": bbox, "vis_bbox": vis_bbox,
                                "height": height, "vis_ratio": vis_ratio}
                    val_gt_json['annotations'].append(ann_json)

        print('New validation set size: {}.'.format(len(val_anns)))

        if not os.path.exists(new_fname_val):
            with open(new_fname_val, 'wb') as f:
                pickle.dump(val_anns, f, protocol=2)

        if not os.path.exists(new_fname_val_json):
            print('Save new JSON validation file.')
            with open(new_fname_val_json, 'w') as f:
                json.dump(val_gt_json, f)

    print('Splitted original annotations from {} to training cache {} and validation cache {}.'.format(orig_path,
                                                                                                       new_fname_train,
                                                                                                       new_fname_val))
