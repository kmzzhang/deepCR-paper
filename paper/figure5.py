import warnings
warnings.filterwarnings("ignore")
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
print('Generating Figure 5: CR mask comparison in resolved galaxy images.')

dtype = torch.cuda.FloatTensor # GPU training
filename = 'data/ACS-WFC-F606W-test.pkl'
dset_test_GAL = data(filename, field='GAL')

gpu_available = torch.cuda.is_available()
if gpu_available:
    model = deepCR(mask='ACS-WFC-F606W-2-32', device='GPU')
else:
    model = deepCR(mask='ACS-WFC-F606W-2-32', device='CPU')

plt.rcParams['figure.dpi'] = 200

fig, ax = plt.subplots(3, 4, sharex=True, sharey=True)
fig.set_size_inches(15 / 1.2 / 6.2 * 4, 3 * 2.5 / 1.2)
plt.subplots_adjust(wspace=0, hspace=0)

idx = [1, 3, 3]
locs = [[230, 165], [240, 110], [241, 198]]

for i in range(3):
    ii = idx[i]
    mask = (1 - dset_test_GAL[ii][2]) * dset_test_GAL[ii][3]
    img0 = dset_test_GAL[ii][0] * (1 - mask) + dset_test_GAL[ii][1] * mask

    mask_deepcr, inpaint_deepcr = model.clean(img0, threshold=0.5)
    mask_deepcr = mask_deepcr.astype(bool)
    mask_lacosmic, _ = lac.detect_cosmics(img0, objlim=5, sigclip=11,
                                           sigfrac=0.3, gain=1, readnoise=5, satlevel=np.inf,
                                           sepmed=False, cleantype='medmask', niter=4)
    
    vmax = np.percentile(np.log10(dset_test_GAL[ii][1]), 99.9) * -1
    vmin = np.percentile(np.log10(dset_test_GAL[ii][1]), 10) * -1

    loc = locs[i]
    imgg = np.log10(img0)[loc[0] - 38:loc[0], loc[1] - 38:loc[1]]
    inpa = np.log10(inpaint_deepcr)[loc[0] - 38:loc[0], loc[1] - 38:loc[1]]
    mask = dset_test_GAL[ii][2][loc[0] - 38:loc[0], loc[1] - 38:loc[1]]
    mask_lacosmic = mask_lacosmic[loc[0] - 38:loc[0], loc[1] - 38:loc[1]]
    mask_deepcr = mask_deepcr[loc[0] - 38:loc[0], loc[1] - 38:loc[1]]

    ax[i, 0].imshow(imgg * -1, cmap='gray', vmin=vmax, vmax=vmin)
    ax[i, 2].imshow(imgg * mask * -1, cmap='gray', vmin=vmax, vmax=vmin)
    plt.subplot(ax[i, 3])
    plt.imshow(imgg * mask_lacosmic * -1, cmap='gray', vmin=vmax, vmax=vmin, alpha=1)
    ax[i, 1].imshow(imgg * mask_deepcr * -1, cmap='gray', vmin=vmax, vmax=vmin)

ax[0, 0].set_title('image')
ax[0, 2].set_title('ground truth')
ax[0, 1].set_title('deepCR')
ax[0, 3].set_title('LACosmic')

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig('figure/figure5.png', fmt='png', bbox_inches='tight')
print('Figure 5 saved to figure/figure5.png')
