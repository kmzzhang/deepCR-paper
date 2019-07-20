import sys
sys.path.append("../deepCR")

import torch
import matplotlib.pyplot as plt
import numpy as np

from deepCR import deepCR
model = deepCR(mask='ACS-WFC-F606W-2-4', device='GPU')

for param in model.maskNet.parameters():
    param.requires_grad = False
    filters = param.cpu().numpy()
    break

fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
fig.set_size_inches(8,8)
for i, f in enumerate(filters):
    im=ax[i//2,i-i//2*2].imshow(f[0], cmap='gray')
plt.subplots_adjust(wspace=0.25, hspace=0.25)
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.tight_layout()
plt.savefig('figure/figureA1_GC4.png', fmt='png', bbox='tight')