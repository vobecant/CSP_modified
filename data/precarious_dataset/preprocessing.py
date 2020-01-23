import json
import os, sys
from PIL import Image
import numpy as np
import pickle

# precarious dataset 960x720px -> need to switch it to 2048x1024px just as Cityscapes -> pad with zeros
# annotation format: class x y w h (in pixels) ... at least it seems so

NAME2IDX = {'person': 1}


def get_shift(img_w, tgt_w, img_h, tgt_h):
    shift_x = (tgt_w - img_w) // 2
    shift_y = (tgt_h - img_h) // 2
    return shift_x, shift_y


def convert_image(img, tgt_w, tgt_h, shift_x, shift_y):
    resized = Image.new("RGB", (tgt_w, tgt_h))
    resized.paste(img, (shift_x, shift_y))
    return resized


def convert_anno(anno_fname, split, order_id, shift_x, shift_y):
    fname = os.path.split(anno_fname)[1]
    image_id = int(fname.split('.')[0])
    img_fname = '{}/{}'.format(split, fname.replace('.txt', '.jpg'))
    boxes = []
    test_anns = []
    with open(anno_fname, 'r') as f:
        anno = f.readlines()[1:]
    for i, ann in enumerate(anno):
        ann = ann.strip('\n').split(' ')
        cls, xywh = NAME2IDX[ann[0]], [int(a) for a in ann[1:5]]
        assert cls == 1
        xywh_shifted = shift_xywh(xywh, shift_x, shift_y)
        bbox = xywh2xyxy(xywh_shifted)
        boxes.append(bbox)
        if split == 'test':
            test_ann = {"id": order_id,
                        "image_id": image_id,
                        "category_id": cls,
                        "iscrowd": 0,
                        "ignore": 0,
                        "bbox": xywh_shifted,
                        "vis_bbox": xywh_shifted,
                        "height": xywh_shifted[-1],
                        "vis_ratio": 1}
            order_id += 1
            test_anns.append(test_ann)

    converted_ann = {'filepath': img_fname,
                     'vis_bboxes': boxes,
                     'bboxes': np.asarray(boxes, dtype=np.int64),
                     'ignoreareas': np.asarray([])}

    return converted_ann, test_anns, order_id


def convert2cityscapes(split_path, tgt_w, tgt_h, tgt_dir, img_ext='.jpg', anno_ext='.txt'):
    images_folder = split_path
    anno_folder = '{}Ano'.format(split_path)
    images_fnames = [f for f in os.listdir(images_folder) if img_ext in f]

    if not os.path.exists(os.path.join(tgt_dir, 'images')):
        os.makedirs(os.path.join(tgt_dir, 'images'))

    converted_anns = []
    test_anns = []
    images = []
    order_id = 0

    for i, im_fname in enumerate(images_fnames, 1):
        image_id = int(im_fname.split('.')[0])
        images.append({"id": image_id, "im_name": im_fname, "height": 1024, "width": 2048})
        img = Image.open(os.path.join(images_folder, im_fname))
        img_w, img_h = img.size
        shift_x, shift_y = get_shift(img_w, tgt_w, img_h, tgt_h)
        img_converted = convert_image(img, tgt_w, tgt_h, shift_x, shift_y)

        img_converted.save(os.path.join(tgt_dir, 'images', im_fname))

        # convert annotation
        anno_fname = os.path.join(anno_folder, im_fname.replace(img_ext, anno_ext))
        anno_converted, test_anns_image, order_id = convert_anno(anno_fname, split, order_id, shift_x,
                                                                 shift_y)
        converted_anns.append(anno_converted)
        test_anns.extend(test_anns_image)
        print('{}/{} images done'.format(i, len(images_fnames)))

    anns_fname = os.path.join(tgt_dir, '{}_annotations'.format(split))
    with open(anns_fname, 'wb') as f:
        pickle.dump(converted_anns, f, protocol=2)
    if split == 'test':
        categories = [{"id": 1, "name": "person"}]
        test_json = {'categories': categories, 'images': images, 'annotations': test_anns}
        with open('{}.json'.format(anns_fname), 'w') as f:
            json.dump(test_json, f)
    print('Annotations saved to {}'.format(anns_fname))
    print('Split "{}" completed!\n'.format(split))


def xywh2xyxy(xywh):
    # Convert bounding box format from [left, top, width, height] to [left, top, right, bottom]
    left, top, width, height = xywh
    right = left + width
    bottom = top + height
    return [left, top, right, bottom]


def shift_xywh(xywh, shift_x, shift_y):
    left, top, width, height = xywh
    left += shift_x
    top += shift_y
    return [left, top, width, height]


if __name__ == '__main__':
    splits = ['train', 'test']
    base_folder = sys.argv[1]
    save_dir = sys.argv[2]
    tgt_w = 2048
    tgt_h = 1024

    for split in splits:
        split_path = os.path.join(base_folder, split)
        convert2cityscapes(split_path, tgt_w=tgt_w, tgt_h=tgt_h, tgt_dir=save_dir)
