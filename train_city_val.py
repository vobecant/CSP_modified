from __future__ import division, print_function
import random
import sys, os
import time
import numpy as np
import cPickle
from keras.utils import generic_utils
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from shutil import copyfile

from keras_csp import config, data_generators
from keras_csp import losses as losses

# parse experiment name
if len(sys.argv) == 1:
    exp_name = ''
    print('No experiment name given. Run with default parameters.')
else:
    exp_name = '_{}'.format(sys.argv[1])
    print("Given experiment name: '{}'.".format(sys.argv[1]))

# get the config parameters
C = config.Config()
C.gpu_ids = '0,1,2,3'
C.onegpu = 2
C.size_train = (640, 1280)
C.init_lr = 2e-4
C.num_epochs = 150
max_nonimproving_epochs = 10
C.offset = True

num_gpu = len(C.gpu_ids.split(','))
batchsize = C.onegpu * num_gpu
os.environ["CUDA_VISIBLE_DEVICES"] = C.gpu_ids

# get the training data
cache_path_train = 'data/cache/cityperson_trainValTest/train_h50{}'.format(exp_name)
with open(cache_path_train, 'rb') as fid:
    train_data = cPickle.load(fid)
print('Loaded training cache from {}'.format(cache_path_train))
num_imgs_train = len(train_data)
random.shuffle(train_data)
print('num of training samples: {}'.format(num_imgs_train))
data_gen_train = data_generators.get_data(train_data, C, batchsize=batchsize, exp_name=exp_name)

# get the validation data
cache_path_val = 'data/cache/cityperson_trainValTest/val'
with open(cache_path_val, 'rb') as fid:
    val_data = cPickle.load(fid)
print('Loaded validation cache from {}'.format(cache_path_val))
num_imgs_val = len(val_data)
random.shuffle(val_data)
print('num of validation samples: {}'.format(num_imgs_val))
data_gen_val = data_generators.get_data_eval(val_data, C, batchsize=batchsize)
n_iter_eval = len(val_data) // batchsize
eval_report_after = n_iter_eval // 10

# define the base network (resnet here, can be MobileNet, etc)
if C.network == 'resnet50':
    from keras_csp import resnet50 as nn

    weight_path = 'data/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

input_shape_img = (C.size_train[0], C.size_train[1], 3)
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

model.load_weights(weight_path, by_name=True)
model_tea.load_weights(weight_path, by_name=True)
print('load weights from {}'.format(weight_path))

if C.offset:
    out_path = 'output/valmodels/city_val/{}/off{}'.format(C.scale, exp_name)
else:
    out_path = 'output/valmodels/city_val/{}/nooff{}'.format(C.scale, exp_name)
if not os.path.exists(out_path):
    os.makedirs(out_path)
res_file = os.path.join(out_path, 'records.txt')

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
best_loss_val = np.Inf
prev_loss_val = np.Inf
nonimproving_epochs = 0
print('Starting training with lr {} and alpha {}'.format(C.init_lr, C.alpha))
start_time = time.time()
total_loss_r, cls_loss_r1, regr_loss_r1, offset_loss_r1, val_loss_history = [], [], [], [], []
for epoch_num in range(C.num_epochs):
    if nonimproving_epochs == max_nonimproving_epochs:
        print('Maximum number of continuous nonimproving epochs reached! Ending training after epoch {}'.format(
            epoch_num))
        break
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1 + add_epoch, C.num_epochs + C.add_epoch))
    while True:
        try:
            X, Y = next(data_gen_train)
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
            if iter_num == epoch_length:
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
                print('Start evaluation.')
                progbar_val = generic_utils.Progbar(n_iter_eval)
                val_completed = False
                val_losses = []
                val_iter_done = 0
                while not val_completed:
                    X, Y, val_completed = next(data_gen_val)
                    val_loss = model.test_on_batch(X, Y)
                    val_losses.append(val_loss)
                    val_iter_done += 1
                    if val_iter_done % n_iter_eval == 0:
                        progbar_val.update(val_iter_done, [('val_loss', np.mean(val_losses))])
                cur_loss_val = np.mean(val_losses)
                val_loss_history.append(cur_loss_val)
                if cur_loss_val < best_loss_val:
                    print('New best validation loss: {} -> {} . '.format(best_loss_val, cur_loss_val), end='')
                    best_loss_val = cur_loss_val
                    val_model_savefile = os.path.join(out_path, 'best_val.hdf5')
                    copyfile(model_savefile, val_model_savefile)
                    print('Saved the network to {}'.format(val_model_savefile))
                else:
                    print('Current validation loss: {}, best validation loss so far: {}'.format(cur_loss_val,
                                                                                                best_loss_val))
                if cur_loss_val > prev_loss_val:
                    nonimproving_epochs += 1
                    print('Validation loss did not improve for {} epochs (max {} nonimproving epochs allowed)'.format(
                        nonimproving_epochs, max_nonimproving_epochs))
                else:
                    nonimproving_epochs = 0
                prev_loss_val = cur_loss_val

                print('Elapsed time: {}'.format(time.time() - start_time))
                iter_num = 0
                start_time = time.time()
                # End of the epoch
                break

        except Exception as e:
            print('Exception: {}'.format(e))
            continue

    records = np.concatenate((np.asarray(total_loss_r).reshape((-1, 1)),
                              np.asarray(cls_loss_r1).reshape((-1, 1)),
                              np.asarray(regr_loss_r1).reshape((-1, 1)),
                              np.asarray(offset_loss_r1).reshape((-1, 1)),
                              np.asarray(val_loss_history).reshape((-1, 1))), axis=-1)
    np.savetxt(res_file, np.array(records), fmt='%.6f')
print('Training complete, exiting.')
