# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('/home/vobecant/PhD/CSP/nightowls_analysis/test_statistics.pkl', 'rb') as f:
    test_statistics = pickle.load(f)

with open('/home/vobecant/PhD/CSP/nightowls_analysis/train_statistics.pkl', 'rb') as f:
    train_statistics = pickle.load(f)

train_heights = train_statistics['heights']
train_visibilities = np.ones_like(train_heights)  # we do not have this info
test_heights_occluded = test_statistics['height_occluded']
test_heights_reasonable = test_statistics['height_reasonable']
test_heights_all = test_heights_occluded + test_heights_reasonable
test_visibilities_occluded = test_statistics['vis_occluded']
test_visibilities_reasonable = test_statistics['vis_reasonable']
test_visibilities_all = test_heights_occluded + test_visibilities_reasonable

# TODO: comparison of all test vs train
plt.figure()
bins = np.arange(50,600,10)
plt.hist(train_heights, bins=bins, alpha=0.5, label='train', density=True)
plt.hist(test_heights_all, bins=bins, alpha=0.5, label='all test', density=True)
plt.title('Heights, all test.')
plt.legend()
plt.savefig('./heights_trn_vs_test_all.png')

# TODO: comparison of occluded test vs train
plt.figure()
bins = np.arange(50,600,10)
plt.hist(train_heights, bins=bins, alpha=0.5, label='train', density=True)
plt.hist(test_heights_occluded, bins=bins, alpha=0.5, label='occ test', density=True)
plt.title('Heights, occluded test.')
plt.legend()
plt.savefig('./heights_trn_vs_test_occ.png')

# TODO: comparison of all test vs train
plt.figure()
bins = np.arange(50,600,10)
plt.hist(train_heights, bins=bins, alpha=0.5, label='train', density=True)
plt.hist(test_heights_reasonable, bins=bins, alpha=0.5, label='reason test', density=True)
plt.title('Heights, reasonable test..')
plt.legend()
plt.savefig('./heights_trn_vs_test_reason.png')