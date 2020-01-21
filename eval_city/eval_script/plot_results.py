from __future__ import print_function
import os, sys
from coco import COCO
from eval_MR_multisetup import COCOeval
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt


def merge_results(exp_name):
    exp_results = {'Reasonable': {}, 'bare': {}, 'partial': {}, 'heavy': {}}

    epoch_offset = 0 if exp_name != '' else 27
    main_path = '../../output/valresults/city/h/off{}'.format(exp_name)
    for epoch in sorted(os.listdir(main_path)):
        print('file: {}'.format(epoch))
        dt_path = os.path.join(main_path, epoch)
        epoch = int(epoch) + epoch_offset
        respath = os.path.join(dt_path, 'results.txt')
        with open(respath, 'r') as f:
            results = f.readlines()
            results = [float(n) for n in results]
        for i, key in enumerate(exp_results.keys()):
            print('{}: {}'.format(key, results[i]))
            exp_results[key][epoch] = results[i]
        print('')

    return exp_results


def plot_results(all_results):
    splits = ['Reasonable', 'bare', 'partial', 'heavy']
    for split in splits:
        for exp_name, exp_results in all_results.items():
            split_result = exp_results[split]
            x, y = list(split_result.keys()), list(split_result.values())
            plt.plot(x, y, label=exp_name)
        plt.title(split)
        plt.savefig(split)
        plt.close()


if __name__ == '__main__':
    experiments = ['', '1P', 'halfP']

    all_results = {}
    for exp_name in experiments:
        key = exp_name if len(exp_name) else 'baseline'
        exp_results = merge_results(exp_name)
        all_results[key] = exp_results

    plot_results(all_results)