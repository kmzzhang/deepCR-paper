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

dtype = torch.cuda.FloatTensor
filename = 'data/ACS-WFC-F606W-test.pkl'
dset_test_EX = data(filename, field='EX')

gpu_available = torch.cuda.is_available()
if gpu_available:
    model = deepCR(mask='ACS-WFC-F606W-2-32', inpaint='ACS-WFC-F606W-3-32', device='GPU')
else:
    model = deepCR(mask='ACS-WFC-F606W-2-32', inpaint='ACS-WFC-F606W-3-32', device='CPU')

print('Creating Figure 3: example images of deepCR mask prediction in extragalactic fields.')

img0=np.zeros((7*256,15*256))
img1=np.zeros((7*256,15*256))
mask=np.zeros((7*256,15*256))
for i in range(105):
    img0[i//15*256:i//15*256+256, (i-i//15*15)*256:(i-i//15*15)*256+256] = dset_test_EX[i+105][0]
    mask_, img1_ = model.clean(dset_test_EX[i+105][0], threshold=0.5)
    mask[i//15*256:i//15*256+256, (i-i//15*15)*256:(i-i//15*15)*256+256] = mask_
    img1[i//15*256:i//15*256+256, (i-i//15*15)*256:(i-i//15*15)*256+256] = img1_
img0=img0[:1024,-1024:]
img1=img1[:1024,-1024:]
mask=mask[:1024,-1024:]

plt.rcParams['figure.dpi'] = 200
fig,ax = plt.subplots(1,2,frameon=False)
fig.set_size_inches(20,10)
vmin=np.log10(np.percentile(img1, 25))*-1
vmax=np.log10(np.percentile(img1, 99.99))*-1
ax[0].imshow(np.log10(img0)*-1, vmin=vmax, vmax=vmin, cmap='gray')
ax[1].imshow(np.log10(img1)*-1, vmin=vmax, vmax=vmin, cmap='gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('figure/figure3a.png', fmt='png', bbox='tight')

ex_examples = []
rows = 8
fig, ax = plt.subplots(3, rows, sharex=True, sharey=True)
fig.set_size_inches(20, 7.67)
plt.subplots_adjust(wspace=0, hspace=0)
locs = [[420, 120], [800, 50], [120, 740], [690, 650], [940, 740], [100, 200], [1000, 420], [900, 130]]
np.random.seed(2)
for i in range(rows):
    loc = locs[i]
    imgg = np.log10(img0[loc[0] - 48:loc[0], loc[1] - 48:loc[1]]) * -1
    ax[0, i].imshow(imgg, cmap='gray', vmin=vmax, vmax=vmin)
    mx = ma.masked_array(imgg, mask=~mask[loc[0] - 48:loc[0], loc[1] - 48:loc[1]].astype(bool))
    ax[1, i].imshow(mx, cmap='gray', vmin=vmax, vmax=vmin)
    ax[2, i].imshow(np.log10(img1[loc[0] - 48:loc[0], loc[1] - 48:loc[1]]) * -1, cmap='gray', vmin=vmax, vmax=vmin)

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.tight_layout()
plt.savefig('figure/figure3b.png', fmt='png', bbox='tight')

print('Top and bottom panel of Figure 3 saved to figure/figure3a.png and figure/figure3b.png.')
