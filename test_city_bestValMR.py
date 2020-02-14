from __future__ import division, print_function
import os, sys
import time
import cPickle
from keras.layers import Input
from keras.models import Model

from eval_city.dt_txt2json import convert_file
from eval_city.eval_script.coco import COCO
from eval_city.eval_script.eval_MR_multisetup import COCOeval
from keras_csp import config, bbox_process
from keras_csp.utilsfunc import *

# parse experiment name
if len(sys.argv) == 1:
    exp_name = ''
else:
    exp_name = '_{}'.format(sys.argv[1])

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
C = config.Config()
C.offset = True
cache_path = 'data/cache/cityperson/val_500'
with open(cache_path, 'rb') as fid:
    val_data = cPickle.load(fid)
num_imgs = len(val_data)
print('num of val samples: {}'.format(num_imgs))

C.size_test = (1024, 2048)
input_shape_img = (C.size_test[0], C.size_test[1], 3)
img_input = Input(shape=input_shape_img)

# define the base network (resnet here, can be MobileNet, etc)
if C.network == 'resnet50':
    from keras_csp import resnet50 as nn
elif C.network == 'mobilenet':
    from keras_csp import mobilenet as nn
else:
    raise NotImplementedError('Not support network: {}'.format(C.network))

# define the network prediction
preds = nn.nn_p3p4p5(img_input, offset=C.offset, num_scale=C.num_scale, trainable=True)
model = Model(img_input, preds)

if C.offset:
    w_path = 'output/valmodels/city_valMR/{}/off{}'.format(C.scale, exp_name)
    out_path = 'output/valresults/city_valMR/{}/off{}'.format(C.scale, exp_name)
if not os.path.exists(out_path):
    os.makedirs(out_path)

weight1 = os.path.join(w_path, 'best_val.hdf5')
res_path = os.path.join(out_path, 'best_val')
if not os.path.exists(res_path):
    os.makedirs(res_path)
print(res_path)
res_file_txt = os.path.join(res_path, 'val_det.txt')

print('load weights from {}'.format(weight1))
model.load_weights(weight1, by_name=True)
res_all = []
start_time = time.time()
for f in range(num_imgs):
    filepath = val_data[f]['filepath']
    images_dir_name = 'images{}/'.format(exp_name if 'base' not in exp_name else '')
    filepath = filepath.replace('images/', images_dir_name)
    img = cv2.imread(filepath)
    x_rcnn = format_img(img, C)
    Y = model.predict(x_rcnn)

    if C.offset:
        boxes = bbox_process.parse_det_offset(Y, C, score=0.1, down=4)
    else:
        boxes = bbox_process.parse_det(Y, C, score=0.1, down=4, scale=C.scale)
    if len(boxes) > 0:
        f_res = np.repeat(f + 1, len(boxes), axis=0).reshape((-1, 1))
        boxes[:, [2, 3]] -= boxes[:, [0, 1]]
        res_all += np.concatenate((f_res, boxes), axis=-1).tolist()
np.savetxt(res_file_txt, np.array(res_all), fmt='%6f')
print(time.time() - start_time)

dt_path = res_path
json_path = convert_file(res_file_txt)

annFile_test = '/home/vobecant/PhD/CSP/eval_city/val_gt.json'
resFile = os.path.join(dt_path, 'val_dt.json')
respath = os.path.join(dt_path, 'results_testSet.txt')
res_file = open(respath, "w")
print('\nResults on the TEST set:')
for id_setup in range(6):
    print('setup {}'.format(id_setup))
    cocoGt = COCO(annFile_test)
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate(id_setup)
    cocoEval.accumulate()
