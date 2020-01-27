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

    main_path = '../../output/valresults/city_val/h/off_{}'.format(exp_name)
    print('Search main path {} for results.')
    if not os.path.exists(main_path):
        print('{} does not exist. Skipping...'.format(main_path))
        return None
    for epoch in sorted(os.listdir(main_path)):
        print('file: {}'.format(epoch))
        dt_path = os.path.join(main_path, epoch)
        if 'best' in epoch:
            continue
        epoch = int(epoch)
        respath = os.path.join(dt_path, 'results.txt')
        if not os.path.exists(respath):
            print('{} does not exist yet = not yet evaluated.'.format(respath))
            continue
        with open(respath, 'r') as f:
            results = f.readlines()
            results = [float(n) for n in results]
        for i, split in enumerate(SPLITS):
            print('{}: {}'.format(split, results[i]))
            exp_results[split][epoch] = results[i]
        print('')

    losses_file = os.path.join(main_path.replace('valresults', 'valmodels'), 'records.txt')
    with open(losses_file, 'r') as f:
        all_losses = [l.split(' ') for l in f.readlines()]
        trn_losses = {i: float(l[0]) for i, l in enumerate(all_losses, 1) if i >= 50}
        val_scores = {i: float(l[4]) for i, l in enumerate(all_losses, 1) if i >= 50}

    exp_results['training loss'] = trn_losses
    exp_results['val_score'] = val_scores

    return exp_results


def plot_results(all_results):
    for split in SPLITS:
        fig, ax = plt.subplots()
        for exp_name, exp_results in all_results.items():
            if exp_results is None:
                continue
            split_result = exp_results[split]
            x, y = list(split_result.keys()), list(split_result.values())
            plt.plot(x, y, label=exp_name)
        plt.legend()
        plt.title(split)
        ax.yaxis.grid()
        ax.set(xlabel='Epoch', ylabel=r'$MR^{-2}$ [%]')
        plt.savefig('{}_val'.format(split))
        plt.close()

    # plot training losses
    split = 'training loss'
    fig, ax = plt.subplots()
    for exp_name, exp_results in all_results.items():
        if exp_results is None:
            continue
        split_result = exp_results[split]
        x, y = list(split_result.keys()), list(split_result.values())
        plt.plot(x, y, label=exp_name)
    plt.legend()
    plt.title(split)
    ax.yaxis.grid()
    ax.set(xlabel='Epoch', ylabel='loss')
    plt.savefig('{}_val'.format(split))
    plt.close()

    # plot validation scores
    split = 'val_score'
    fig, ax = plt.subplots()
    for exp_name, exp_results in all_results.items():
        if exp_results is None:
            continue
        split_result = exp_results[split]
        x, y = list(split_result.keys()), list(split_result.values())
        plt.plot(x, y, label=exp_name)
    plt.legend()
    plt.title('validation score')
    ax.yaxis.grid()
    ax.set(xlabel='Epoch', ylabel='loss')
    plt.savefig('{}_val'.format(split))
    plt.close()


if __name__ == '__main__':
    experiments = ['baseline', '1P', 'halfP', 'blurred']
    fname = 'results.pkl'

    all_results = {}
    for exp_name in experiments:
        key = exp_name if len(exp_name) else 'baseline'
        exp_results = merge_results(exp_name)
        all_results[key] = exp_results

    with open(fname, 'wb') as f:
        cPickle.dump(all_results, f)

    plot_results(all_results)
