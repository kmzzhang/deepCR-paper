import sys
sys.path.append("../deepCR")
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
import torch
from torch import from_numpy
from skimage.restoration import inpaint_biharmonic
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool
from deepCR import deepCR
from data import data
from util import *

filename = 'data/ACS-WFC-F606W-test.pkl'
dset_test_EX = data(filename, field='EX', augment=1, random_nmask=False)
dset_test_GC = data(filename, field=['GC1','GC2'], augment=1, random_nmask=False)
dset_test_GAL = data(filename, field='GAL', augment=1, random_nmask=False)

def test_EX(inputs):
    fn, rand = inputs
    mse_arr = []
    for s in rand:
        np.random.seed(s)
        mse=0
        indexes = np.random.randint(0,len(dset_test_EX),size=200)
        for i in indexes:
            img0, img1, mask, badmask, _, _ = dset_test_EX[i]
            out = fn(img0, mask)
            out = out*(1-badmask)
            img0=img0*(1-badmask)
            mse += (((out-img0)*mask)**2).sum()/mask.sum()
        mse/=i+1
        mse_arr.append(mse)
    return mse_arr

def test_GC(inputs):
    fn, rand = inputs
    mse_arr = []
    for s in rand:
        np.random.seed(s)
        mse=0
        indexes = np.random.randint(0,len(dset_test_GC),size=200)
        for i in indexes:
            img0, img1, mask, badmask, _, _ = dset_test_GC[i]
            out = fn(img0, mask)
            out = out*(1-badmask)
            img0=img0*(1-badmask)
            mse += (((out-img0)*mask)**2).sum()/mask.sum()
        mse/=i+1
        mse_arr.append(mse)
    return mse_arr

def test_GAL(inputs):
    fn, rand=inputs
    mse_arr = []
    for s in rand:
        np.random.seed(s)
        mse=0
        indexes = np.random.randint(0,len(dset_test_GAL),size=200)
        for i in indexes:
            img0, img1, mask, badmask, _, _ = dset_test_GAL[i]
            out = fn(img0, mask)
            out = out*(1-badmask)
            img0=img0*(1-badmask)
            mse += (((out-img0)*mask)**2).sum()/mask.sum()

            #if (((out-img0)*mask)**2).sum()/mask.sum()>100:
            #    print(i,(((out-img0)*mask)**2).sum()/mask.sum())
        mse/=i+1
        mse_arr.append(mse)
    return mse_arr


if os.path.isfile('inpaint_non_neural.npy'):
    print('non neural metrics loaded from inpaint_non_neural.npy')
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    save = np.load('inpaint_non_neural.npy')
    np.load = np_load_old
    med_ex = save[0]
    med_gal = save[1]
    med_gc = save[2]
    bi_ex = save[3]
    bi_gal = save[4]
    bi_gc = save[5]

else:
    print('calculating non neural metrics')
    with Pool(12) as p:
        med_ex = p.map(test_EX, [[medmask, np.arange(i*3,(i+1)*3)] for i in range(16)])
    with Pool(12) as p:
        med_gal = p.map(test_GAL, [[medmask, np.arange(i * 3, (i + 1) * 3)] for i in range(16)])
    with Pool(12) as p:
        med_gc = p.map(test_GC, [[medmask, np.arange(i * 3, (i + 1) * 3)] for i in range(16)])
    with Pool(12) as p:
        bi_gal = p.map(test_GAL, [[inpaint_biharmonic, np.arange(i * 3, (i + 1) * 3)] for i in range(16)])
    with Pool(12) as p:
        bi_ex = p.map(test_EX, [[inpaint_biharmonic, np.arange(i*3,(i+1)*3)] for i in range(16)])
    with Pool(12) as p:
        bi_gc = p.map(test_GC, [[inpaint_biharmonic, np.arange(i*3,(i+1)*3)] for i in range(16)])

    med_ex = np.array(med_ex).reshape(-1)
    med_gal = np.array(med_gal).reshape(-1)
    med_gc = np.array(med_gc).reshape(-1)
    bi_ex = np.array(bi_ex).reshape(-1)
    bi_gal = np.array(bi_gal).reshape(-1)
    bi_gc = np.array(bi_gc).reshape(-1)
    results = np.array([med_ex, med_gal, med_gc, bi_ex, bi_gal, bi_gc])
    np.save('inpaint_non_neural.npy', results)
    print('non neural metrics saved to inpaint_non_neural.npy')

if os.path.isfile('inpaint_deepCR.npy'):
    print('deepCR metrics loaded from inpaint_deepCR.npy')
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    save = np.load('inpaint_deepCR.npy')
    np.load = np_load_old
    deep_ex = save[0]
    deep_gal = save[1]
    deep_gc = save[2]
    deep4_ex = save[3]
    deep4_gal = save[4]
    deep4_gc = save[5]
    model = deepCR(inpaint='ACS-WFC-F606W-3-32', device='GPU')
    modelcpu = deepCR(inpaint='ACS-WFC-F606W-3-32', device='CPU')
    model2 = deepCR(inpaint='ACS-WFC-F606W-2-32', device='GPU')
    model2cpu = deepCR(inpaint='ACS-WFC-F606W-2-32', device='CPU')

else:
    print('calculating deepCR metrics')
    model = deepCR(inpaint='ACS-WFC-F606W-3-32', device='GPU')
    modelcpu = deepCR(inpaint='ACS-WFC-F606W-3-32', device='CPU')

    deep_ex = test_EX((model.inpaint, range(48)))
    deep_gal = test_GAL((model.inpaint, range(48)))
    deep_gc = test_GC((model.inpaint, range(48)))

    model2 = deepCR(inpaint='ACS-WFC-F606W-2-32', device='GPU')
    model2cpu = deepCR(inpaint='ACS-WFC-F606W-2-32', device='CPU')
    deep4_ex = test_EX((model2.inpaint, range(48)))
    deep4_gal = test_GAL((model2.inpaint, range(48)))
    deep4_gc = test_GC((model2.inpaint, range(48)))

    deep_ex = np.array(deep_ex).reshape(-1)
    deep_gal = np.array(deep_gal).reshape(-1)
    deep_gc = np.array(deep_gc).reshape(-1)
    deep4_ex = np.array(deep4_ex).reshape(-1)
    deep4_gal = np.array(deep4_gal).reshape(-1)
    deep4_gc = np.array(deep4_gc).reshape(-1)
    results = np.array([deep_ex, deep_gal, deep_gc, deep4_ex, deep4_gal, deep4_gc])
    np.save('inpaint_deepCR.npy', results)
    print('deepCR metrics saved to inpaint_deepCR.npy')

np.random.seed(0)
t0 = time.time()
for i in range(100):
    a=medmask(dset_test_EX[i][0], dset_test_EX[i][2])
t1 = time.time()
med_time = t1-t0

np.random.seed(0)
t0 = time.time()
for i in range(1):
    a=inpaint_biharmonic(dset_test_EX[i][0], dset_test_EX[i][2])
t1 = time.time()
bi_time = (t1-t0) * 100

dtype = torch.cuda.FloatTensor
img0 = from_numpy(dset_test_EX[0][0]).type(dtype).reshape(1,1,256,256)
mask = from_numpy(dset_test_EX[0][2]).type(dtype).reshape(1,1,256,256)
var = torch.cat([img0,mask], dim=1)
img0cpu = from_numpy(dset_test_EX[0][0]).type(torch.FloatTensor).reshape(1,1,256,256)
maskcpu = from_numpy(dset_test_EX[0][2]).type(torch.FloatTensor).reshape(1,1,256,256)
varcpu = torch.cat([img0cpu,maskcpu], dim=1)

t0 = time.time()
for i in range(100):
    a=model.inpaintNet(var)
t1 = time.time()
deep_time = (t1-t0)

t0 = time.time()
for i in range(100):
    a=modelcpu.inpaintNet(varcpu)
t1 = time.time()
deep_time_cpu = t1-t0

t0 = time.time()
for i in range(100):
    a=model2.inpaintNet(var)
t1 = time.time()
deep4_time = (t1-t0)


t0 = time.time()
for i in range(100):
    a=model2cpu.inpaintNet(varcpu)
t1 = time.time()
deep4_time_cpu = t1-t0

time_cpu={'deepCR-2-32': deep4_time_cpu,
         'deepCR-3-32': deep_time_cpu,
         'medmask': med_time,
         'biharmonic': bi_time}

time_gpu={'deepCR-2-32': deep4_time,
         'deepCR-3-32': deep_time,
         'medmask': 0,
         'biharmonic': 0}

ex_flux={'deepCR-2-32': deep4_ex,
         'deepCR-3-32': deep_ex,
         'medmask': med_ex,
         'biharmonic': bi_ex}

gal_flux={'deepCR-2-32': deep4_gal,
          'deepCR-3-32': deep_gal,
          'medmask': med_gal,
          'biharmonic': bi_gal}

gc_flux={'deepCR-2-32': deep4_gc,
          'deepCR-3-32': deep_gc,
          'medmask': med_gc,
          'biharmonic': bi_gc}

table3 = []
table3.append('\\begin{table*}\n \\label{table:inpaint}\n \\caption{Image inpainting metrics of deepCR-inpaint as compared to baseline models. MSE is average mean squared error of 48 test iterations, each iteration containing 200 image stamps and their respectively inpainting mask randomly sampled from the test set. The last two columns shows runtime, evaluated either on 4 Intel Xeon CPUs, or 4 Nvidia Tesla GPUs, and only included network forward pass time for deepCR.}\n \\centering\n')
table3.append(
    '  \\begin{tabular}{llllll} \n \\toprule \n         & extragalactic    & globular cluster & resolved galaxy & \\multicolumn{2}{c}{runtime}\\\\\n        Model & MSE & MSE'
    '& MSE & CPU & GPU  \\\\\n \\midrule\n')
for i in ex_flux.keys():
    temp = '  ' + i + '&'
    med = np.array(ex_flux[i]).mean()
    temp += '%.3f & ' % med
    med = np.array(gc_flux[i]).mean()
    temp += '%.3f & ' % med
    med = np.array(gal_flux[i]).mean()
    temp += '%.3f & ' % med

    if (time_cpu[i] != 0):
        temp += '%.1f & ' % (time_cpu[i])
    else:
        temp += 'n/a &'
    if (time_gpu[i] != 0):
        temp += '%.1f  ' % (time_gpu[i])
    else:
        temp += 'n/a '
    temp += '\\\\\n'
    table3.append(temp)
table3.append('\\bottomrule \n \\end{tabular}\n\\end{table*}')

with open("table/table3.txt", "w") as file:
    for line in table3:
        file.write(line)