import json
import sys, os

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
    new_fname_train = 'train_{}'.format(exp_name)
    new_fname_val = 'val'
    new_fname_val_json = 'val_gt.json'

    # step 1
    # create new training split from the EXTENDED original training split
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

    # step 2
    # create new validation split from the NONEXTENDED original training split
    if not os.path.exists(new_fname_val) or not os.path.exists(new_fname_val_json):
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
                    val_gt_json['images'].append({'id': len(images) + 1,
                                                  'im_name': image_name,
                                                  'height': 1024, 'width': 2048})
                for bbox, vis_bbox in zip(ann['bboxes'], ann['vis_bboxes']):
                    vis_ratio = (vis_bbox[-2] * vis_bbox[-1]) / (bbox[-2] * bbox[-1])
                    ann_json = {"id": len(val_gt_json['annotations']) + 1, "image_id": images_added[image_name],
                                "category_id": 1, "iscrowd": 0, "ignore": 0,
                                "bbox": bbox, "vis_bbox": vis_bbox,
                                "height": bbox[-1], "vis_ratio": vis_ratio}
                    val_gt_json['annotations'].append(ann_json)

        print('New validation set size: {}.'.format(len(val_anns)))

        if not os.path.exists(new_fname_val):
            with open(new_fname_val, 'wb') as f:
                pickle.dump(val_anns, f, protocol=2)

        if not os.path.exists(new_fname_val_json):
            with open(new_fname_val_json, 'w') as f:
                json.dump(val_gt_json,f)

    print('Splitted original annotations from {} to training cache {} and validation cache {}.'.format(orig_path,
                                                                                                       new_fname_train,
                                                                                                       new_fname_val))
