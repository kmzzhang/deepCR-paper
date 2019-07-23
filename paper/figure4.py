import os
# Change the number of GPU in use as you wish
#os.environ['CUDA_VISIBLE_DEVICES']="1"
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

print('Creating Figure 4: ROC curves')

gpu_available = torch.cuda.is_available()
if not gpu_available:
    print('-----------------------------------------------------')
    print('Warning: GPU not detected on your device.')
    print('It is fine if you are generating figures and tables')
    print('    from the pre-computed data in the *.npy files')
    print('Otherwise, deepCR benchmark takes very long to run on CPU.')
    print('We recommend that you abort and switch to device with GPU,')
    print('    if you would like to run the benchmarks from stretch.')
    print('-----------------------------------------------------')

# load test data
filename = 'data/ACS-WFC-F606W-test.pkl'
dset_test_EX = data(filename, field='EX')
dset_test_GC = data(filename, field='GC1')
dset_test_GAL = data(filename, field='GAL')

# first calculate ROC curves for LACosmic
# to save time, we saved calculated
# LACosmic ROC curves in LAC.npy
# if LAC.npy doesn't exist we will calculate again
# variables ending in _3 are with rad-3 square dilation
if os.path.isfile('ROC_LACosmic.npy'):
    save = np.load('ROC_LACosmic.npy')
    TPR_EX_LAC = save[0]
    FPR_EX_LAC = save[1]
    TPR_EX_LAC_3 = save[2]
    FPR_EX_LAC_3 = save[3]
    TPR_GAL_LAC = save[4]
    FPR_GAL_LAC = save[5]
    TPR_GAL_LAC_3 = save[6]
    FPR_GAL_LAC_3 = save[7]
    TPR_GC_LAC = save[8]
    FPR_GC_LAC = save[9]
    TPR_GC_LAC_3 = save[10]
    FPR_GC_LAC_3 = save[11]
    print('Loaded LACosmic ROC curves from ROC_LACosmic.npy')
else:
    print('calculating LACosmic ROC curves')
    (TPR_EX_LAC, FPR_EX_LAC),(TPR_EX_LAC_3, FPR_EX_LAC_3) = ROC_LACosmic(dset_test_EX, np.linspace(3,30,100), 2, limit=320, dilate=square(3))
    (TPR_GAL_LAC, FPR_GAL_LAC), (TPR_GAL_LAC_3, FPR_GAL_LAC_3) = ROC_LACosmic(dset_test_GAL, np.linspace(4,60,100), 5, limit=320, dilate=square(3))
    (TPR_GC_LAC, FPR_GC_LAC), (TPR_GC_LAC_3, FPR_GC_LAC_3) = ROC_LACosmic(dset_test_GC,
                                                                          np.linspace(2 ** (1 / 4), 40 ** (1 / 4),
                                                                                      100) ** 4, 3.5,
                                                                          limit=320, dilate=square(3))

    save = []
    for i in [TPR_EX_LAC, FPR_EX_LAC, TPR_EX_LAC_3, FPR_EX_LAC_3, TPR_GAL_LAC, FPR_GAL_LAC, TPR_GAL_LAC_3,
              FPR_GAL_LAC_3, TPR_GC_LAC, FPR_GC_LAC, TPR_GC_LAC_3, FPR_GC_LAC_3]:
        save.append(i)
    np.save('ROC_LACosmic.npy', np.array(save))
    print('saved to ROC_LACosmic.npy')

if os.path.isfile('ROC_deepCR.npy'):
    save = np.load('ROC_deepCR.npy')
    TPR_EX = save[0]
    FPR_EX = save[1]
    TPR_EX_3 = save[2]
    TPR_GAL = save[3]
    FPR_GAL = save[4]
    TPR_GAL_3 = save[5]
    TPR_EX4 = save[6]
    FPR_EX4 = save[7]
    TPR_EX4_3 = save[8]
    TPR_GAL4 = save[9]
    FPR_GAL4 = save[10]
    TPR_GAL4_3 = save[11]
    TPR_GC = save[12]
    FPR_GC = save[13]
    TPR_GC_3 = save[14]
    TPR_GC4 = save[15]
    FPR_GC4 = save[16]
    TPR_GC4_3 = save[17]
    if gpu_available:
        deepCR_2_32 = deepCR(mask='ACS-WFC-F606W-2-32', device='GPU')
        deepCR_2_4 = deepCR(mask='ACS-WFC-F606W-2-4', device='GPU')
    print('Loaded deepCR ROC curves from ROC_deepCR.npy')
else:
    # calculate ROC curves for two variants of deepCR-mask
    print('calculating deepCR ROC curves.')
    if gpu_available:
        deepCR_2_32 = deepCR(mask='ACS-WFC-F606W-2-32', device='GPU')
        deepCR_2_4 = deepCR(mask='ACS-WFC-F606W-2-4', device='GPU')
    else:
        deepCR_2_32 = deepCR(mask='ACS-WFC-F606W-2-32', device='CPU')
        deepCR_2_4 = deepCR(mask='ACS-WFC-F606W-2-4', device='CPU')
    (TPR_EX, FPR_EX), (TPR_EX_3, FPR_EX_3) = ROC_DAE(deepCR_2_32, dset_test_EX, np.linspace(0.001, 0.999, 500), 320,
                                                     dilate=square(3))
    (TPR_GAL, FPR_GAL), (TPR_GAL_3, FPR_GAL_3) = ROC_DAE(deepCR_2_32, dset_test_GAL, np.linspace(0.01, 0.99, 500), 320,
                                                         dilate=square(3))

    (TPR_EX4, FPR_EX4), (TPR_EX4_3, FPR_EX4_3) = ROC_DAE(deepCR_2_4, dset_test_EX, np.linspace(0.001, 0.999, 500), 320,
                                                         dilate=square(3))
    (TPR_GAL4, FPR_GAL4), (TPR_GAL4_3, FPR_GAL4_3) = ROC_DAE(deepCR_2_4, dset_test_GAL, np.linspace(0.01, 0.99, 500),
                                                             320, dilate=square(3))
    (TPR_GC, FPR_GC), (TPR_GC_3, FPR_GC_3) = ROC_DAE(deepCR_2_32, dset_test_GC,
                                                     np.linspace(0.0001 ** (1 / 10), 0.999 ** (1 / 10), 500) ** 10, 320,
                                                     dilate=square(3))
    (TPR_GC4, FPR_GC4), (TPR_GC4_3, FPR_GC4_3) = ROC_DAE(deepCR_2_4, dset_test_GC,
                                                         np.linspace(0.0001 ** (1 / 10), 0.999 ** (1 / 10), 500) ** 10,
                                                         320,
                                                         dilate=square(3))
    save = []
    for i in [TPR_EX, FPR_EX, TPR_EX_3, TPR_GAL, FPR_GAL, TPR_GAL_3, TPR_EX4, FPR_EX4, TPR_EX4_3, TPR_GAL4, FPR_GAL4,
              TPR_GAL4_3, TPR_GC, FPR_GC, TPR_GC_3, TPR_GC4, FPR_GC4, TPR_GC4_3]:
        save.append(i)
    np.save('ROC_deepCR.npy', np.array(save))
    print('saved to ROC_deepCR.npy')

# generate figure 3
plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(FPR_EX, TPR_EX_3, 'r--', label='deepCR-2-32 +', linewidth=1.8)
plt.plot(FPR_EX_LAC, TPR_EX_LAC_3, 'r:', label='LACosmic +', linewidth=1.2)
plt.plot(FPR_EX, TPR_EX, 'k-', label='deepCR-2-32', linewidth=1.8)
plt.plot(FPR_EX_LAC, TPR_EX_LAC, 'k-.', label='LACosmic', linewidth=1.2)

plt.legend(loc=4)
plt.xlim(0,1)
plt.ylim(40,100)
plt.xlabel('false positive rate [%]', fontsize=12)
plt.ylabel('true positive rate [%]', fontsize=12)
plt.title('extragalactic field', fontsize=12)

plt.subplot(132)
plt.plot(FPR_GC, TPR_GC_3, 'r--', label='deepCR-2-32 +', linewidth=1.8)
plt.plot(FPR_GC_LAC, TPR_GC_LAC_3, 'r:', label='LACosmic +', linewidth=1.2)
plt.plot(FPR_GC, TPR_GC, 'k-', label='deepCR-2-32', linewidth=1.8)
plt.plot(FPR_GC_LAC, TPR_GC_LAC, 'k-.', label='LACosmic', linewidth=1.2)

plt.legend(loc=4)
plt.xlim(0,1)
plt.ylim(40,100)
plt.xlabel('false positive rate [%]', fontsize=12)
plt.ylabel('true positive rate [%]', fontsize=12)
plt.title('globular cluster', fontsize=12)

plt.subplot(133)
plt.plot(FPR_GAL, TPR_GAL_3, 'r--', label='deepCR-2-32 +', linewidth=1.8)
plt.plot(FPR_GAL_LAC, TPR_GAL_LAC_3, 'r:', label='LACosmic +', linewidth=1.2)
plt.plot(FPR_GAL, TPR_GAL, 'k-', label='deepCR-2-32', linewidth=1.8)
plt.plot(FPR_GAL_LAC, TPR_GAL_LAC, 'k-.', label='LACosmic', linewidth=1.2)

plt.legend(loc=4)
plt.xlim(0,1)
plt.ylim(40,100)
plt.xlabel('false positive rate [%]', fontsize=12)
plt.title('resolved galaxy', fontsize=12)
plt.tight_layout()
plt.savefig('figure/ROC.pdf', fmt='pdf', bbox_inches='tight')

print('Figure 4 saved to figure/ROC.pdf')

# interpolate f:FPR --> TPR
fex = interp1d(FPR_EX,TPR_EX)
fex_lac = interp1d(FPR_EX_LAC,TPR_EX_LAC)
fex3 = interp1d(FPR_EX,TPR_EX_3)
fex3_lac = interp1d(FPR_EX_LAC,TPR_EX_LAC_3)

fex4 = interp1d(FPR_EX4,TPR_EX4)
fex43 = interp1d(FPR_EX4,TPR_EX4_3)
fgal4 = interp1d(FPR_GAL4,TPR_GAL4)
fgal43 = interp1d(FPR_GAL4,TPR_GAL4_3)


fgal = interp1d(FPR_GAL,TPR_GAL)
fgal_lac = interp1d(FPR_GAL_LAC,TPR_GAL_LAC)
fgal3 = interp1d(FPR_GAL,TPR_GAL_3)
fgal3_lac = interp1d(FPR_GAL_LAC,TPR_GAL_LAC_3)

fgc3 = interp1d(FPR_GC,TPR_GC_3)
fgc3_lac = interp1d(FPR_GC_LAC,TPR_GC_LAC_3)
fgc = interp1d(FPR_GC, TPR_GC)
fgc_lac = interp1d(FPR_GC_LAC,TPR_GC_LAC)
fgc4 = interp1d(FPR_GC4,TPR_GC4)
fgc43 = interp1d(FPR_GC4,TPR_GC4_3)

# dummy variable to evaluate runtime on deepCR
if gpu_available:
    var = from_numpy(dset_test_EX[0][0]).type(torch.cuda.FloatTensor).reshape(1,1,256,256)
    t0= time.time()
    for i in range(500):
        a=deepCR_2_32.maskNet(var)
    deepCR_GPU = (time.time()-t0)/5
    t0= time.time()
    for i in range(500):
        deepCR_2_4.maskNet(var)
    deepCR4_GPU = (time.time()-t0)/5
else:
    deepCR_GPU = 0
    deepCR4_GPU = 0

varcpu = from_numpy(dset_test_EX[0][0]).type(torch.FloatTensor).reshape(1,1,256,256)
deepCR_2_32_cpu = deepCR(mask='ACS-WFC-F606W-2-32', device='CPU')
t0= time.time()
for i in range(500):
    a=deepCR_2_32_cpu.maskNet(varcpu)
deepCR_CPU = (time.time()-t0)/5

deepCR_2_4_cpu = deepCR(mask='ACS-WFC-F606W-2-4', device='CPU')
t0= time.time()
for i in range(500):
    deepCR_2_4_cpu.maskNet(varcpu)
deepCR4_CPU = (time.time()-t0)/5
t0= time.time()
for i in range(500):
    a = lac.detect_cosmics(dset_test_EX[0][0], objlim=5, sigclip=2,
                   sigfrac=0.3, gain=1, readnoise=5, satlevel = np.inf,
                   sepmed=False, cleantype='medmask', niter=4)
LACosmic_CPU = (time.time()-t0)/5

# generate table2
table2 = []
table2.append('\\begin{table*}[t]\n \\label{table:mask}\n \\caption{}\n \\centering\n')
table2.append('  \\begin{tabular}{*{6}{c}} \n \\toprule \n          & extragalactic field & globular cluster & resolved galaxy & \\multicolumn{2}{c}{runtime}   \\\\\n        Model &TPR (0.05\%)& TPR (0.05\%)& TPR (0.05\%)& CPU & GPU \\\\\n \\midrule\n')

ans='deepCR-2-4 &'
ans+=str(fex4(0.05))[:4]+'\% ('+str(fex43(0.05))[:4]+'\%) &'
ans+=str(fgc4(0.05))[:4]+'\% ('+str(fgc43(0.05))[:4]+'\%) &'
ans+=str(fgal4(0.05))[:4]+'\% ('+str(fgal43(0.05))[:4]+'\%) &'
ans+='\\textbf{%.1fs} &'%deepCR4_CPU
ans+='\\textbf{%.1fs}'%deepCR4_GPU
ans+='\\\\\n'
table2.append(ans)
ans='deepCR-2-32 &'
ans+='\\textbf{'+str(fex(0.05))[:4]+'\%} ('+str(fex3(0.05))[:4]+'\%) &'
ans+='\\textbf{'+str(fgc(0.05))[:4]+'\%} ('+str(fgc3(0.05))[:4]+'\%)&'
ans+='\\textbf{'+str(fgal(0.05))[:4]+'\%} ('+str(fgal3(0.05))[:4]+'\%)&'
ans+='%.1fs &'%deepCR_CPU
ans+='%.1fs'%deepCR_GPU
ans+='\\\\\n'
table2.append(ans)

ans='\\LACosmic &'
ans+=str(fex_lac(0.05))[:4]+'\% ('+str(fex3_lac(0.05))[:4]+'\%) &'
ans+=str(fgc_lac(0.05))[:4]+'\% ('+str(fgc3_lac(0.05))[:4]+'\%) &'
ans+=str(fgal_lac(0.05))[:4]+'\% ('+str(fgal3_lac(0.05))[:4]+'\%) &'
ans+='%.1fs'%LACosmic_CPU + ' & n/a'
ans+='\\\\\n'
table2.append(ans)

table2.append(' \midrule \n \midrule \n &TPR (0.5\%)& TPR (0.5\%)& TPR (0.5\%) \\\\\n \\midrule \n')

ans='deepCR-2-4 &'
ans+=str(fex4(0.5))[:4]+'\% ('+str(fex43(0.5))[:4]+'\%) &'
ans+=str(fgc4(0.5))[:4]+'\% ('+str(fgc43(0.5))[:4]+'\%) &'
ans+=str(fgal4(0.5))[:4]+'\% ('+str(fgal43(0.5))[:4]+'\%) &'
ans+='\\\\\n'
table2.append(ans)

ans='deepCR-2-32 &'
ans+='\\textbf{'+str(fex(0.5))[:4]+'\%} ('+str(fex3(0.5))[:4]+'\%) &'
ans+='\\textbf{'+str(fgc(0.5))[:4]+'\%} ('+str(fgc3(0.5))[:4]+'\%)&'
ans+='\\textbf{'+str(fgal(0.5))[:4]+'\%} ('+str(fgal3(0.5))[:4]+'\%)&'
ans+='\\\\\n'
table2.append(ans)

ans='\\LACosmic &'
ans+=str(fex_lac(0.5))[:4]+'\% ('+str(fex3_lac(0.5))[:4]+'\%) &'
ans+=str(fgc_lac(0.5))[:4]+'\% ('+str(fgc3_lac(0.5))[:4]+'\%) &'
ans+=str(fgal_lac(0.5))[:4]+'\% ('+str(fgal3_lac(0.5))[:4]+'\%) &'
ans+='\\\\\n'
table2.append(ans)


table2.append('\\bottomrule \n \\end{tabular}\n\\end{table*}')
with open("table/table2.txt", "w") as file:
    for line in table2:
        file.write(line)
print('Table 2 saved to table/table2.txt')
