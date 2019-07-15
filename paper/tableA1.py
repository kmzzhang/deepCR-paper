import sys
sys.path.append("..")
import os
import time
import torch
import torch.nn as nn
from torch import from_numpy
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import dilation, square
from scipy.interpolate import interp1d

import astroscrappy.astroscrappy as lac
from deepCR import deepCR

from util import ROC_LACosmic, ROC_DAE, maskMetric
from data import data
dtype = torch.cuda.FloatTensor # GPU training

filename = 'data/ACS-WFC-F606W-test.pkl'
dset_test_EX = data(filename, field='EX')
dset_test_GAL = data(filename, field='GAL')

model = deepCR(mask='ACS-WFC-F606W-2-32', device='GPU')
(TPR_EX, FPR_EX), (TPR_EX_3, FPR_EX_3) = ROC_DAE(model, dset_test_EX, np.linspace(0.001,0.999,100), 320)
(TPR_GAL, FPR_GAL), (TPR_GAL_3 , FPR_GAL_3) = ROC_DAE(model, dset_test_GAL, np.linspace(0.01,0.99,100), 320)
model = deepCR(mask='ACS-WFC-F606W-2-4', device='GPU')
(TPR_EX4, FPR_EX4), (TPR_EX4_3, FPR_EX4_3) = ROC_DAE(model, dset_test_EX, np.linspace(0.001,0.999,100), 20)
(TPR_GAL4, FPR_GAL4), (TPR_GAL4_3 , FPR_GAL4_3) = ROC_DAE(model, dset_test_GAL, np.linspace(0.01,0.99,100), 20)

dset_test_EX = data(filename, field='EX', aug_sky=[-0.1,3])
dset_test_GAL = data(filename, field='GAL', aug_sky=[-0.1,3])

model = deepCR(mask='ACS-WFC-F606W-2-32', device='GPU')
(TPR_EX_, FPR_EX_), (TPR_EX_3_, FPR_EX_3_) = ROC_DAE(model, dset_test_EX, np.linspace(0.001,0.999,100), 320)
(TPR_GAL_, FPR_GAL_), (TPR_GAL_3_ , FPR_GAL_3_) = ROC_DAE(model, dset_test_GAL, np.linspace(0.01,0.99,100), 320)
model = deepCR(mask='ACS-WFC-F606W-2-4', device='GPU')
(TPR_EX4_, FPR_EX4_), (TPR_EX4_3_, FPR_EX4_3_) = ROC_DAE(model, dset_test_EX, np.linspace(0.001,0.999,100), 20)
(TPR_GAL4_, FPR_GAL4_), (TPR_GAL4_3_ , FPR_GAL4_3_) = ROC_DAE(model, dset_test_GAL, np.linspace(0.01,0.99,100), 20)

"""
    Table A1
"""
fex = interp1d(FPR_EX,TPR_EX)
fex4 = interp1d(FPR_EX4,TPR_EX4)
fgal4 = interp1d(FPR_GAL4,TPR_GAL4)
fgal = interp1d(FPR_GAL,TPR_GAL)

fex_ = interp1d(FPR_EX_,TPR_EX_)
fex4_ = interp1d(FPR_EX4_,TPR_EX4_)
fgal4_ = interp1d(FPR_GAL4_,TPR_GAL4_)
fgal_ = interp1d(FPR_GAL_,TPR_GAL_)

table2 = []
table2.append('\\begin{table*}\n \\label{table:mask}\n \\caption{tbd}\n \\centering\n')
table2.append('  \\begin{tabular}{lll|ll|ll} \n \\toprule \n          & t_{min} & t_{max} & sparse-exgal & & dense-gal& & &    \\\\\n        Model &TPR (0.02\%) & TPR (0.1\%)& TPR (0.02\%)&TPR (0.1\%) \\\\\n \\midrule\n')

ans='deepCR-2-4 & 400s & 540s &'
ans+=str(fex4(0.02))[:4]+'\% &'
ans+=str(fex4(0.1))[:4]+'\%  &'
ans+=str(fgal4(0.02))[:4]+'\% &'
ans+=str(fgal4(0.1))[:4]+'\% &'
ans+='\\\\\n'
table2.append(ans)

ans='deepCR-2-4 & 40s & 2160s &'
ans+=str(fex4_(0.02))[:4]+'\% &'
ans+=str(fex4_(0.1))[:4]+'\%  &'
ans+=str(fgal4_(0.02))[:4]+'\% &'
ans+=str(fgal4_(0.1))[:4]+'\% &'
ans+='\\\\\n \\midrule \n'
table2.append(ans)

ans='deepCR-2-32 & 400s & 540s &'
ans+=str(fex(0.02))[:4]+'\% &'
ans+=str(fex(0.1))[:4]+'\%  &'
ans+=str(fgal(0.02))[:4]+'\%&'
ans+=str(fgal(0.1))[:4]+'\% &'
ans+='\\\\\n'
table2.append(ans)


ans='deepCR-2-32 & 40s & 2160s &'
ans+=str(fex_(0.02))[:4]+'\% &'
ans+=str(fex_(0.1))[:4]+'\%  &'
ans+=str(fgal_(0.02))[:4]+'\%&'
ans+=str(fgal_(0.1))[:4]+'\% &'
ans+='\\\\\n'
table2.append(ans)


table2.append('\\bottomrule \n \\end{tabular}\n\\end{table*}')
print('saved')
with open("table/tableA1.txt", "w") as file:
    for line in table2:
        file.write(line)