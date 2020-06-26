# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('/home/vobecant/PhD/CSP/cityperson_analysis/test_statistics.pkl', 'rb') as f:
    test_statistics = pickle.load(f)

with open('/home/vobecant/PhD/CSP/cityperson_analysis/train_statistics.pkl', 'rb') as f:
    train_statistics = pickle.load(f)

train_heights = train_statistics['heights']
train_visibilities = train_statistics['visibilities']
test_heights_occluded = test_statistics['height_occluded']
test_heights_reasonable = test_statistics['height_reasonable']
test_heights_all = test_heights_occluded + test_heights_reasonable
test_visibilities_occluded = test_statistics['vis_occluded']
test_visibilities_reasonable = test_statistics['vis_reasonable']
test_visibilities_all = test_heights_occluded + test_visibilities_reasonable

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist_trn, xedges_trn, yedges_trn = np.histogram2d(train_heights, train_visibilities, bins=[
    np.arange(50, max(max(train_heights), max(test_heights_all)), 10), np.arange(0, 1.0, 0.05)], density=True)
hist_tst, xedges_tst, yedges_tst = np.histogram2d(test_heights_all, test_visibilities_all, bins=[
    np.arange(50, max(max(train_heights), max(test_heights_all)), 10), np.arange(0, 1.0, 0.05)], density=True)

'''
#training positions
xpos_trn, ypos_trn = np.meshgrid(xedges_trn[:-1] + 0.025, yedges_trn[:-1] + 0.025, indexing="ij")
xpos_trn = xpos_trn.ravel()
ypos_trn = ypos_trn.ravel()
zpos_trn = 0
# Construct arrays with the dimensions for the bars.
dx_trn = dy_trn = 0.5 * np.ones_like(zpos_trn)
dz_trn = hist_trn.ravel()

# test positions
xpos_tst, ypos_tst = np.meshgrid(xedges_tst[:-1] - 0.025, yedges_tst[:-1] - 0.025, indexing="ij")
xpos_tst = xpos_tst.ravel()
ypos_tst = ypos_tst.ravel()
zpos_tst = 0
# Construct arrays with the dimensions for the bars.
dx_tst = dy_tst = 0.5 * np.ones_like(zpos_tst)
dz_tst = hist_trn.ravel()

ax.bar3d(xpos_trn, ypos_trn, zpos_trn, dx_trn, dy_trn, dz_trn, zsort='average', color='blue')
ax.bar3d(xpos_tst, ypos_tst, zpos_tst, dx_tst, dy_tst, dz_tst, zsort='average', color='red')
plt.savefig('./test_vs_train.jpg')
'''

x_trn, y_trn = np.meshgrid(xedges_trn[:-1], yedges_trn[:-1], indexing="ij")
z_trn = hist_trn
ax.plot_wireframe(x_trn, y_trn, z_trn, color='blue', label='all train')

x_tst, y_tst = np.meshgrid(xedges_tst[:-1], yedges_tst[:-1], indexing="ij")
z_tst = hist_tst
ax.plot_wireframe(x_tst, y_tst, z_tst, color='red', label='all test')
ax.legend()

ax.set_title('All train vs all test')

plt.savefig('./test_all_vs_train.jpg')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist_tst_r, xedges_tst_r, yedges_tst_r = np.histogram2d(test_heights_reasonable, test_visibilities_reasonable, bins=[
    np.arange(50, max(max(train_heights), max(test_heights_reasonable)), 10), np.arange(0, 1.0, 0.05)], density=True)
ax.plot_wireframe(x_trn, y_trn, z_trn, color='blue', label='all train')

x_tst, y_tst = np.meshgrid(xedges_tst_r[:-1], yedges_tst_r[:-1], indexing="ij")
z_tst = hist_tst_r
ax.plot_wireframe(x_tst, y_tst, z_tst, color='red', label='reasonable test')
ax.legend()
ax.set_title('All train vs reasonable test')
plt.savefig('./test_reason_vs_train.jpg')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist_tst_o, xedges_tst_o, yedges_tst_o = np.histogram2d(test_heights_occluded, test_visibilities_occluded, bins=[
    np.arange(50, max(max(train_heights), max(test_heights_occluded)), 10), np.arange(0, 1.0, 0.05)], density=True)
ax.plot_wireframe(x_trn, y_trn, z_trn, color='blue', label='all train')

x_tst, y_tst = np.meshgrid(xedges_tst_o[:-1], yedges_tst_o[:-1], indexing="ij")
z_tst = hist_tst_o
ax.plot_wireframe(x_tst, y_tst, z_tst, color='red', label='occluded test')
ax.legend()
ax.set_title('All train vs occluded test')
plt.savefig('./test_occ_vs_train.jpg')
plt.close()
