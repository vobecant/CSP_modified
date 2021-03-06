from __future__ import division, print_function
import random
import sys, os
import time

import cv2
import numpy as np
import cPickle
from keras.utils import generic_utils
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from shutil import copyfile

from eval_utils import *
from keras_csp import config, data_generators, bbox_process
from keras_csp import losses as losses

# parse experiment name
from keras_csp.utilsfunc import format_img

if len(sys.argv) == 1:
    exp_name = ''
    print('No experiment name given. Run with default parameters.')
else:
    exp_name = '_{}'.format(sys.argv[1])
    print("Given experiment name: '{}'.".format(sys.argv[1]))

# get the config parameters
C = config.Config()
C_tst = config.TestConfig()
if len(sys.argv) == 3:
    debug = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    C.gpu_ids = '0'
    print(sys.argv)
else:
    debug = False
    C.gpu_ids = '0,1,2,3'

EVAL = False
C.onegpu = 2
C.size_train = (640, 1280)
C_tst.size_test = (640, 1280)
C_tst.size_train = (640, 1280)
C.init_lr = 2e-4
C.num_epochs = 150
max_nonimproving_epochs = 5
C.offset = True

num_gpu = len(C.gpu_ids.split(','))
batchsize = C.onegpu * num_gpu
print('num_gpu: {}, batchsize: {}, C.gpu_ids: {}'.format(num_gpu, batchsize, C.gpu_ids))
os.environ["CUDA_VISIBLE_DEVICES"] = C.gpu_ids

# get the training data
cache_path_train = 'data/cache/cityperson_trainValTest/train_h50{}'.format(exp_name)
with open(cache_path_train, 'rb') as fid:
    train_data = cPickle.load(fid)
print('Loaded training cache from {}'.format(cache_path_train))
num_imgs_train = len(train_data)
random.shuffle(train_data)
print('num of training samples: {}'.format(num_imgs_train))
data_gen_train = data_generators.get_data_eccv(train_data, C, batchsize=batchsize, exp_name=exp_name)

# get the validation data
cache_path_val = 'data/cache/cityperson_trainValTest/val'
annFile = 'data/cityperson_trainValTest/val_gt.json'
with open(cache_path_val, 'rb') as fid:
    val_data = cPickle.load(fid)
annFile = '/home/vobecant/PhD/CSP/data/cache/cityperson_trainValTest/val_gt_fullRes.json'
img_id_lut = json.load(open(annFile, 'r'))['images']
img_id_lut = {tmp['im_name']: tmp['id'] for tmp in img_id_lut}
data_gen_val = data_generators.get_data_eval_eccv(val_data, C_tst, batchsize=batchsize, exp_name=exp_name,
                                                  return_fname=True)

# define the base network (resnet here, can be MobileNet, etc)
if C.network == 'resnet50':
    from keras_csp import resnet50 as nn

    weight_path = 'data/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

input_shape_img = (None, None, 3)
img_input = Input(shape=input_shape_img)
# define the network prediction
preds = nn.nn_p3p4p5(img_input, offset=C.offset, num_scale=C.num_scale, trainable=True)
preds_tea = nn.nn_p3p4p5(img_input, offset=C.offset, num_scale=C.num_scale, trainable=True)

model = Model(img_input, preds)
if num_gpu > 1:
    from keras_csp.parallel_model import ParallelModel

    model = ParallelModel(model, int(num_gpu))
    model_stu = Model(img_input, preds)
model_tea = Model(img_input, preds_tea)

load_st = time.time()
model.load_weights(weight_path, by_name=True)
model_tea.load_weights(weight_path, by_name=True)
print('load weights from {} in {}'.format(weight_path, time.time() - load_st))

if C.offset:
    out_path = 'output/valmodels/city_valMR_eccv/{}/off{}'.format(C.scale, exp_name)
else:
    out_path = 'output/valmodels/city_valMR_eccv/{}/nooff{}'.format(C.scale, exp_name)
print('Output path: {}'.format(out_path))
if not os.path.exists(out_path):
    os.makedirs(out_path)
res_file_all = os.path.join(out_path, 'records.txt')

optimizer = Adam(lr=C.init_lr)
if C.offset:
    model.compile(optimizer=optimizer, loss=[losses.cls_center, losses.regr_h, losses.regr_offset])
else:
    if C.scale == 'hw':
        model.compile(optimizer=optimizer, loss=[losses.cls_center, losses.regr_hw])
    else:
        model.compile(optimizer=optimizer, loss=[losses.cls_center, losses.regr_h])

epoch_length = int(C.iter_per_epoch / batchsize)
iter_num = 0
add_epoch = 0
losses = np.zeros((epoch_length, 3))

best_loss_train = np.Inf
best_mr_val = np.Inf
prev_mr_val = np.Inf
nonimproving_epochs = 0
print('Starting training with lr {} and alpha {}'.format(C.init_lr, C.alpha))
start_time = time.time()
total_loss_r, cls_loss_r1, regr_loss_r1, offset_loss_r1 = [], [], [], []
val_mr_history = []
for epoch_num in range(C.num_epochs):
    if nonimproving_epochs == max_nonimproving_epochs:
        print('Maximum number of continuous nonimproving epochs reached! Ending training after epoch {}'.format(
            epoch_num))
        break
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1 + add_epoch, C.num_epochs + C.add_epoch))
    while True:
        # try:
        X, Y = next(data_gen_train)
        if iter_num == 0:
            print('Training. X.shape={}, Y.shape={}'.format(X[0].shape, Y[0].shape))
        loss_s1 = model.train_on_batch(X, Y)

        for l in model_tea.layers:
            weights_tea = l.get_weights()
            if len(weights_tea) > 0:
                if num_gpu > 1:
                    weights_stu = model_stu.get_layer(name=l.name).get_weights()
                else:
                    weights_stu = model.get_layer(name=l.name).get_weights()
                weights_tea = [C.alpha * w_tea + (1 - C.alpha) * w_stu for (w_tea, w_stu) in
                               zip(weights_tea, weights_stu)]
                l.set_weights(weights_tea)
        # print loss_s1
        losses[iter_num, 0] = loss_s1[1]
        losses[iter_num, 1] = loss_s1[2]
        if C.offset:
            losses[iter_num, 2] = loss_s1[3]
        else:
            losses[iter_num, 2] = 0

        iter_num += 1
        if iter_num % 20 == 0:
            progbar.update(iter_num,
                           [('cls', np.mean(losses[:iter_num, 0])), ('regr_h', np.mean(losses[:iter_num, 1])),
                            ('offset', np.mean(losses[:iter_num, 2]))])
        if iter_num == epoch_length:  # or len(sys.argv) == 3:
            cls_loss1 = np.mean(losses[:, 0])
            regr_loss1 = np.mean(losses[:, 1])
            offset_loss1 = np.mean(losses[:, 2])
            total_loss = cls_loss1 + regr_loss1 + offset_loss1

            total_loss_r.append(total_loss)
            cls_loss_r1.append(cls_loss1)
            regr_loss_r1.append(regr_loss1)
            offset_loss_r1.append(offset_loss1)
            print('Total loss: {}'.format(total_loss))

            if total_loss < best_loss_train:
                print('Total loss decreased from {} to {}, saving weights'.format(best_loss_train, total_loss))
                best_loss_train = total_loss
            model_savefile = os.path.join(out_path,
                                          'net_e{}_l{}.hdf5'.format(epoch_num + 1 + add_epoch, total_loss))
            model_tea.save_weights(model_savefile)

            # validate the model
            if EVAL:
                print('Start evaluation.')
                # TODO: compute MR-2 for validation on *REASONABLE* subset
                # 1) run equivalent of test_city.py with the latest network
                start_time_val = time.time()
                res_path = os.path.join(out_path, 'valDt_ep{}'.format(epoch_num))
                if not os.path.exists(res_path):
                    os.makedirs(res_path)
                res_file = os.path.join(res_path, 'val_det.txt')
                print('Save validation detections to {}'.format(res_file))
                res_all = []
                val_completed = False
                while not val_completed:
                    X, val_completed, fnames = next(data_gen_val)
                    Y = model.predict(X)
                    if C.offset:
                        boxes_batch = bbox_process.parse_det_offset_batch(Y, C_tst, score=0.1, down=4)
                    else:
                        boxes_batch = bbox_process.parse_det(Y, C_tst, score=0.1, down=4, scale=C.scale)
                    # boxes are in XYXY format
                    for boxes, fname in zip(boxes_batch, fnames):
                        if len(boxes) > 0:
                            img_id = img_id_lut[fname]
                            f_res = np.repeat(img_id, len(boxes), axis=0).reshape((-1, 1))
                            # change boxes to XYWH format
                            boxes[:, [2, 3]] -= boxes[:, [0, 1]]
                            res_all += np.concatenate((f_res, boxes), axis=-1).tolist()
                np.savetxt(res_file, np.array(res_all), fmt='%6f')

                # 2) transform detections from .txt file to JSON
                json_filepath = convert_file(res_file)

                # 3) run evaluation for the REASONABLE subset
                cur_mr_val_reasonable = eval_json_reasonable(annFile, json_filepath)

                # 4) compare to the best reasonable MR-2 so far
                val_mr_history.append(cur_mr_val_reasonable)
                if cur_mr_val_reasonable < best_mr_val:
                    print('New best validation MR-2: {} -> {} . '.format(best_mr_val, cur_mr_val_reasonable), end='')
                    best_mr_val = cur_mr_val_reasonable
                    val_model_savefile = os.path.join(out_path, 'best_val.hdf5')
                    copyfile(model_savefile, val_model_savefile)
                    print('Saved the network to {}'.format(val_model_savefile))
                else:
                    print('Current validation MR-2: {}, best validation MR-2 so far: {}'.format(cur_mr_val_reasonable,
                                                                                                best_mr_val))
                if cur_mr_val_reasonable > prev_mr_val:
                    nonimproving_epochs += 1
                    print('Validation MR-2 did not improve for {} epochs (max {} nonimproving epochs allowed)'.format(
                        nonimproving_epochs, max_nonimproving_epochs))
                else:
                    print('Validation MR-2 better than in the previous step. Current: {}, previous: {}'.format(
                        cur_mr_val_reasonable, prev_mr_val))
                    nonimproving_epochs = 0
                prev_mr_val = cur_mr_val_reasonable

                print('Elapsed time for validation: {}'.format(time.time() - start_time_val))
            print('Elapsed time for epoch: {}'.format(time.time() - start_time))

            iter_num = 0
            start_time = time.time()
            # End of the epoch
            break

            # except Exception as e:
            #    print('Exception: {}'.format(e))
            #    continue

    records = np.concatenate((np.asarray(total_loss_r).reshape((-1, 1)),
                              np.asarray(cls_loss_r1).reshape((-1, 1)),
                              np.asarray(regr_loss_r1).reshape((-1, 1)),
                              np.asarray(offset_loss_r1).reshape((-1, 1))), axis=-1)
    # np.asarray(val_mr_history).reshape((-1, 1))), axis=-1)
    np.savetxt(res_file_all, np.array(records), fmt='%.6f')
print('Training complete, exiting.')
