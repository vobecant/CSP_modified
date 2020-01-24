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
    exp_name = os.path.split(orig_path)[1].split('_')[-1]
    new_fname_train = 'train_{}'.format(exp_name)
    new_fname_val = 'val'

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
    if not os.path.exists(new_fname_val):
        with open('../cityperson/train_h50', 'rb') as f:
            if py == 3:
                orig_train = pickle.load(f, encoding='latin1')
            else:
                orig_train = pickle.load(f)

        val_anns = []
        for ann in orig_train:
            city = get_city(ann['filepath'])
            if city in VAL_CITIES:
                val_anns.append(ann)

        print('New validation set size: {}.'.format(len(val_anns)))

        with open(new_fname_val, 'wb') as f:
            pickle.dump(val_anns, f, protocol=2)

    print('Splitted original annotations from {} to training cache {} and validation cache {}.'.format(orig_path,
                                                                                                       new_fname_train,
                                                                                                       new_fname_val))
