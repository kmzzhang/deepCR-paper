import sys
sys.path.append("../deepCR")

import torch
import matplotlib.pyplot as plt
import numpy as np
from deepCR import deepCR

print('Generating Figure 6: convolution kernels')

model = deepCR(mask='example_model', device='CPU')

for param in model.maskNet.parameters():
    param.requires_grad = False
    filters = param.cpu().numpy()
    break

fig, ax = plt.subplots(4,8, sharex=True, sharey=True)
fig.set_size_inches(16,8)
for i, f in enumerate(filters):
    im = ax[i//8, i-i//8*8].imshow(f[0], cmap='gray')
plt.subplots_adjust(wspace=0.25, hspace=0.25)
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.tight_layout()
plt.savefig('figure/figure6.png', fmt='png', bbox='tight')

print('Figure saved to figure/figure6.png')
