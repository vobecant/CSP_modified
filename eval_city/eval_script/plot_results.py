from __future__ import print_function
import os
import cPickle
from coco import COCO
from eval_MR_multisetup import COCOeval
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SPLITS = ['Reasonable', 'Reasonable_small', 'bare', 'partial', 'heavy', 'All']


def merge_results(exp_name):
    exp_results = {split: {} for split in SPLITS}

    epoch_offset = 0 if exp_name != '' else 27
    main_path = '../../output/valresults/city/h/off_{}'.format(exp_name)
    if not os.path.exists(main_path):
        print('{} does not exist. Skipping...'.format(main_path))
        return None
    for epoch in sorted(os.listdir(main_path)):
        print('file: {}'.format(epoch))
        dt_path = os.path.join(main_path, epoch)
        epoch = int(epoch) + epoch_offset
        respath = os.path.join(dt_path, 'results.txt')
        if not os.path.exists(respath):
            print('{} does not exist yet = not yet evaluated.'.format(respath))
            continue
        with open(respath, 'r') as f:
            results = f.readlines()
            results = [float(n) for n in results]
        for i, split in enumerate(exp_results.keys()):
            print('{}: {}'.format(split, results[i]))
            exp_results[split][epoch] = results[i]
        print('')

    return exp_results


def plot_results(all_results):
    for split in SPLITS:
        for exp_name, exp_results in all_results.items():
            if exp_results is None:
                continue
            split_result = exp_results[split]
            x, y = list(split_result.keys()), list(split_result.values())
            plt.plot(x, y, label=exp_name)
        plt.legend()
        plt.title(split)
        plt.savefig(split)
        plt.close()


if __name__ == '__main__':
    experiments = ['baseline', '1P', 'halfP', 'ganonymized', 'blurred']
    fname = 'results.pkl'

    all_results = {}
    for exp_name in experiments:
        key = exp_name if len(exp_name) else 'baseline'
        exp_results = merge_results(exp_name)
        all_results[key] = exp_results

    with open(fname, 'wb') as f:
        cPickle.dump(all_results, f)

    print([s for s in SPLITS])
    plot_results(all_results)
