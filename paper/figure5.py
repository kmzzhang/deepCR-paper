import sys
sys.path.append("../deepCR")

import torch
from torch import from_numpy
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import astroscrappy.astroscrappy as lac
from deepCR import deepCR
from data import data

dtype = torch.cuda.FloatTensor # GPU training
filename = 'data/ACS-WFC-F606W-test.pkl'
dset_test_GAL = data(filename, field='GAL')

model = deepCR(mask='ACS-WFC-F606W-2-32', device='GPU')

plt.rcParams['figure.dpi'] = 200
rows_gal = 3
rows = rows_gal

fig, ax = plt.subplots(rows, 4, sharex=True, sharey=True)
fig.set_size_inches(15 / 1.2 / 6.2 * 4, rows * 2.5 / 1.2)
plt.subplots_adjust(wspace=0, hspace=0)

#idx = [1, 2, 3, 3, 5]
#locs = [[230, 165], [75, 230], [240, 110], [241, 198], [125, 38]]
idx = [1, 3, 3, 5]
locs = [[230, 165], [240, 110], [241, 198], [125, 38]]

for i in range(rows_gal):
    ii = idx[i]
    mask = (1 - dset_test_GAL[ii][2]) * dset_test_GAL[ii][3]
    img0 = dset_test_GAL[ii][0] * (1 - mask) + dset_test_GAL[ii][1] * mask

    mask_deepcr, inpaint_deepcr = model.clean(img0, threshold=0.5)
    mask_deepcr = mask_deepcr.astype(bool)
    #mask_lacosmic, _ = lac.detect_cosmics(img0 * 100, objlim=5, sigclip=20,
    #                                      sigfrac=0.3, gain=1, readnoise=5, satlevel=np.inf,
    #                                      sepmed=False, cleantype='medmask', niter=4)
    mask_lacosmic2, _ = lac.detect_cosmics(img0 * 100, objlim=5, sigclip=11,
                                           sigfrac=0.3, gain=1, readnoise=5, satlevel=np.inf,
                                           sepmed=False, cleantype='medmask', niter=4)
    
    vmax = np.percentile(np.log10(dset_test_GAL[ii][1]), 99.9) * -1
    vmin = np.percentile(np.log10(dset_test_GAL[ii][1]), 10) * -1

    loc = locs[i]
    imgg = np.log10(img0)[loc[0] - 38:loc[0], loc[1] - 38:loc[1]]
    inpa = np.log10(inpaint_deepcr)[loc[0] - 38:loc[0], loc[1] - 38:loc[1]]
    mask = dset_test_GAL[ii][2][loc[0] - 38:loc[0], loc[1] - 38:loc[1]]
    #mask_lacosmic = mask_lacosmic[loc[0] - 38:loc[0], loc[1] - 38:loc[1]]
    mask_lacosmic2 = mask_lacosmic2[loc[0] - 38:loc[0], loc[1] - 38:loc[1]]
    mask_deepcr = mask_deepcr[loc[0] - 38:loc[0], loc[1] - 38:loc[1]]

    ax[i, 0].imshow(imgg * -1, cmap='gray', vmin=vmax, vmax=vmin)
    ax[i, 2].imshow(imgg * mask * -1, cmap='gray', vmin=vmax, vmax=vmin)
    plt.subplot(ax[i, 3])
    plt.imshow(imgg * mask_lacosmic2 * -1, cmap='gray', vmin=vmax, vmax=vmin, alpha=1)
    #plt.imshow(imgg * mask_lacosmic * -1, cmap='gray', vmin=vmax, vmax=vmin, alpha=0.8)
    ax[i, 1].imshow(imgg * mask_deepcr * -1, cmap='gray', vmin=vmax, vmax=vmin)

ax[0, 0].set_title('image')
ax[0, 2].set_title('ground truth')
ax[0, 1].set_title('deepCR')
ax[0, 3].set_title('LACosmic')

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig('figure/figure5.png', fmt='png', bbox_inches='tight')