import sys
sys.path.append("..")
from deepCR import deepCR
import numpy as np
model = deepCR(mask='ACS-WFC-F606W-2-32', device='GPU')
input = np.ones((256,256))
out = model.clean(input)
print(out[0].shape, out[1].shape)