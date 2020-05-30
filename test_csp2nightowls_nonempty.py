from __future__ import division, print_function
import os, sys
import time
import cPickle
from keras.layers import Input
from keras.models import Model
from keras_csp import config, bbox_process
from keras_csp.utilsfunc import *

w_path = sys.argv[1]
weights = os.path.join(w_path,'for_no.hdf5')
out_path = w_path.replace('valmodels', 'valresults')
if not os.path.exists(out_path):
    os.makedirs(out_path)

C = config.Config()
C.offset = True
cache_path = 'data/cache/nightowls/val_h50_nonempty_xyxy'
with open(cache_path, 'rb') as fid:
    val_data = cPickle.load(fid)
num_imgs = len(val_data)
print('num of val samples: {}'.format(num_imgs))

C.size_test = (640, 1024)
# image channel-wise mean to subtract, the order is BGR!!!
C.img_channel_mean = [62.22632229, 76.71838384, 78.05682094]
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


res_path = os.path.join(out_path, 'no')
if not os.path.exists(res_path):
    os.makedirs(res_path)
print(res_path)
res_file = os.path.join(res_path, 'val_det_no.txt')
print('load weights from {}'.format(weights))
model.load_weights(weights, by_name=True)
res_all = []
start_time = time.time()
for f in range(num_imgs):
    filepath = val_data[f]['filepath']
    image_id = val_data[f]['image_id']
    img = cv2.imread(filepath)
    x_rcnn = format_img(img, C)
    Y = model.predict(x_rcnn)

    if C.offset:
        boxes = bbox_process.parse_det_offset(Y, C, score=0.1, down=4)
    else:
        boxes = bbox_process.parse_det(Y, C, score=0.1, down=4, scale=C.scale)
    if len(boxes) > 0:
        f_res = np.repeat(image_id, len(boxes), axis=0).reshape((-1, 1))
        boxes[:, [2, 3]] -= boxes[:, [0, 1]]
        res_all += np.concatenate((f_res, boxes), axis=-1).tolist()
np.savetxt(res_file, np.array(res_all), fmt='%6f')
print(time.time() - start_time)