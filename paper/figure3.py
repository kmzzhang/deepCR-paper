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

if os.path.isfile('LAC.npy'):
    save = np.load('LAC.npy')
    TPR_EX_LAC = save[0]
    FPR_EX_LAC = save[1]
    TPR_EX_LAC_3 = save[2]
    FPR_EX_LAC_3 = save[3]
    TPR_GAL_LAC = save[4]
    FPR_GAL_LAC = save[5]
    TPR_GAL_LAC_3 = save[6]
    FPR_GAL_LAC_3 = save[7]
    print('loaded')
else:
    (TPR_EX_LAC, FPR_EX_LAC),(TPR_EX_LAC_3, FPR_EX_LAC_3) = ROC_LACosmic(dset_test_EX, np.linspace(3,30,100), 2.6, limit=320, dilate=square(3))
    (TPR_GAL_LAC, FPR_GAL_LAC), (TPR_GAL_LAC_3, FPR_GAL_LAC_3) = ROC_LACosmic(dset_test_GAL, np.linspace(6,60,100), 4.5, limit=320, dilate=square(3))
    save = []
    for i in [TPR_EX_LAC, FPR_EX_LAC,TPR_EX_LAC_3, FPR_EX_LAC_3, TPR_GAL_LAC, FPR_GAL_LAC, TPR_GAL_LAC_3, FPR_GAL_LAC_3]:
        save.append(i)
    np.save('LAC.npy', np.array(save))

model = deepCR(mask='ACS-WFC-F606W-2-32', device='GPU')
(TPR_EX, FPR_EX), (TPR_EX_3, FPR_EX_3) = ROC_DAE(model, dset_test_EX, np.linspace(0.001,0.999,100), 320,  dilate=square(3))
(TPR_GAL, FPR_GAL), (TPR_GAL_3 , FPR_GAL_3) = ROC_DAE(model, dset_test_GAL, np.linspace(0.01,0.99,100), 320, dilate=square(3))
model = deepCR(mask='ACS-WFC-F606W-2-4', device='GPU')
(TPR_EX4, FPR_EX4), (TPR_EX4_3, FPR_EX4_3) = ROC_DAE(model, dset_test_EX, np.linspace(0.001,0.999,100), 20,  dilate=square(3))
(TPR_GAL4, FPR_GAL4), (TPR_GAL4_3 , FPR_GAL4_3) = ROC_DAE(model, dset_test_GAL, np.linspace(0.01,0.99,100), 20, dilate=square(3))

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(FPR_EX, TPR_EX_3, 'r-', label='deepCR-2-32 +', linewidth=2,alpha=0.5)
plt.plot(FPR_EX_LAC, TPR_EX_LAC_3, 'r--', label='LACosmic +', linewidth=2,alpha=0.5)

plt.plot(FPR_EX, TPR_EX, 'k-', label='deepCR-2-32', linewidth=1)
plt.plot(FPR_EX_LAC, TPR_EX_LAC, 'k-.', label='LACosmic', linewidth=1)

plt.legend(loc=4)
plt.xlim(0,1)
plt.ylim(40,100)
plt.xlabel('false positive rate [%]', fontsize=12)
plt.ylabel('true positive rate [%]', fontsize=12)
plt.title('sparse-exgal', fontsize=12)


plt.subplot(122)
plt.plot(FPR_GAL, TPR_GAL_3, 'r-', label='deepCR-2-32 +', linewidth=2,alpha=0.5)
plt.plot(FPR_GAL_LAC, TPR_GAL_LAC_3, 'r--', label='LACosmic +', linewidth=2,alpha=0.5)

plt.plot(FPR_GAL, TPR_GAL, 'k-', label='deepCR-2-32', linewidth=1)
plt.plot(FPR_GAL_LAC, TPR_GAL_LAC, 'k-.', label='LACosmic', linewidth=1)

plt.legend(loc=4)
plt.xlim(0,1)
plt.ylim(40,100)
plt.xlabel('false positive rate [%]', fontsize=12)
plt.title('dense-gal', fontsize=12)
plt.tight_layout()
plt.savefig('figure/ROC.pdf', fmt='pdf', bbox_inches='tight')

"""
    Table 2
"""
fex = interp1d(FPR_EX,TPR_EX)
fex_lac = interp1d(FPR_EX_LAC,TPR_EX_LAC)
fex3 = interp1d(FPR_EX,TPR_EX_3)
fex3_lac = interp1d(FPR_EX,TPR_EX_LAC_3)

fex4 = interp1d(FPR_EX4,TPR_EX4)
fex43 = interp1d(FPR_EX4,TPR_EX4_3)
fgal4 = interp1d(FPR_GAL4,TPR_GAL4)
fgal43 = interp1d(FPR_GAL4,TPR_GAL4_3)


fgal = interp1d(FPR_GAL,TPR_GAL)
fgal_lac = interp1d(FPR_GAL_LAC,TPR_GAL_LAC)
fgal3 = interp1d(FPR_GAL,TPR_GAL_3)
fgal3_lac = interp1d(FPR_GAL,TPR_GAL_LAC_3)

var = from_numpy(dset_test_EX[0][0]).type(dtype).reshape(1,1,256,256)
varcpu = from_numpy(dset_test_EX[0][0]).type(torch.FloatTensor).reshape(1,1,256,256)

model = deepCR(mask='ACS-WFC-F606W-2-32', device='GPU')
t0= time.time()
for i in range(10000):
    a=model.maskNet(var)
deepCR_GPU = (time.time()-t0)/100

modelcpu = deepCR(mask='ACS-WFC-F606W-2-32', device='CPU')
t0= time.time()
for i in range(2000):
    a=modelcpu.maskNet(varcpu)
deepCR_CPU = (time.time()-t0)/20

model = deepCR(mask='ACS-WFC-F606W-2-4', device='GPU')
t0= time.time()
for i in range(10000):
    model.maskNet(var)
deepCR4_GPU = (time.time()-t0)/100

modelcpu = deepCR(mask='ACS-WFC-F606W-2-4', device='CPU')
t0= time.time()
for i in range(2000):
    modelcpu.maskNet(varcpu)
deepCR4_CPU = (time.time()-t0)/20

t0= time.time()
for i in range(500):
    a = lac.detect_cosmics(dset_test_EX[0][0]*100, objlim=5.5, sigclip=2,
                   sigfrac=0.3, gain=1, readnoise=5, satlevel = np.inf,
                   sepmed=False, cleantype='medmask', niter=6)
LACosmic_CPU = (time.time()-t0)/5

table2 = []
table2.append('\\begin{table*}\n \\label{table:mask}\n \\caption{Mask prediction benchmarks. Each column shows true positive rates (TPR) at the specified false positive rates (FPR; 0.01\% and 0.1\%), for either sparse-exgal or dense-gal fields. TPR in parenthesis shows benchmark value after mask dilation, as described in the main text.}\n \\centering\n')
table2.append('  \\begin{tabular}{l|ll|ll|ll} \n \\toprule \n          &sparse-exgal & & dense-gal& & &    \\\\\n        Model &TPR (0.02\%) & TPR (0.1\%)& TPR (0.02\%)&TPR (0.1\%)&Time (CPU) &Time (GPU)  \\\\\n \\midrule\n')

ans='deepCR-2-4 &'
ans+=str(fex4(0.02))[:4]+'\% ('+str(fex43(0.02))[:4]+'\%) &'
ans+=str(fex4(0.1))[:4]+'\% ('+str(fex43(0.1))[:4]+'\%) &'
ans+=str(fgal4(0.02))[:4]+'\% ('+str(fgal43(0.02))[:4]+'\%) &'
ans+=str(fgal4(0.1))[:4]+'\% ('+str(fgal43(0.1))[:4]+'\%) &'
ans+='\\textbf{%.1f} &'%deepCR4_CPU
ans+='\\textbf{%.1f}'%deepCR4_GPU
ans+='\\\\\n'
table2.append(ans)

ans='deepCR-2-32 &'
ans+='\\textbf{'+str(fex(0.02))[:4]+'\%} ('+str(fex3(0.02))[:4]+'\%) &'
ans+='\\textbf{'+str(fex(0.1))[:4]+'\%} ('+str(fex3(0.1))[:4]+'\%) &'
ans+='\\textbf{'+str(fgal(0.02))[:4]+'\%} ('+str(fgal3(0.02))[:4]+'\%)&'
ans+='\\textbf{'+str(fgal(0.1))[:4]+'\%} ('+str(fgal3(0.1))[:4]+'\%)&'
ans+='%.1f &'%deepCR_CPU
ans+='%.1f'%deepCR_GPU
ans+='\\\\\n'
table2.append(ans)

ans='LACosmic &'
ans+=str(fex_lac(0.02))[:4]+'\% ('+str(fex3_lac(0.02))[:4]+'\%) &'
ans+=str(fex_lac(0.1))[:4]+'\% ('+str(fex3_lac(0.1))[:4]+'\%) &'
ans+=str(fgal_lac(0.02))[:4]+'\% ('+str(fgal3_lac(0.02))[:4]+'\%) &'
ans+=str(fgal_lac(0.1))[:4]+'\% ('+str(fgal3_lac(0.1))[:4]+'\%) &'
ans+='%.1f'%LACosmic_CPU + ' & -'
ans+='\\\\\n'
table2.append(ans)

table2.append('\\bottomrule \n \\end{tabular}\n\\end{table*}')
print('saved')
with open("table/table2.txt", "w") as file:
    for line in table2:
        file.write(line)