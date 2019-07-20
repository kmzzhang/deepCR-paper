import numpy as np
import astroscrappy.astroscrappy as lac
import matplotlib.pyplot as plt
from skimage.morphology import dilation, disk

def maskMetric(PD, GT):
    if len(PD.shape) == 2:
        PD = PD.reshape(1, *PD.shape)
    if len(GT.shape) == 2:
        GT = GT.reshape(1, *GT.shape)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(GT.shape[0]):
        P = GT[i].sum()
        TP += (PD[i][GT[i] == 1] == 1).sum()
        TN += (PD[i][GT[i] == 0] == 0).sum()
        FP += (PD[i][GT[i] == 0] == 1).sum()
        FN += (PD[i][GT[i] == 1] == 0).sum()
    return (np.array([TP, TN, FP, FN]))

def ROC_DAE(model, dset, threshold, limit=1e5, dilate=None):
    np.random.seed(0)
    nROC = threshold.size
    metric = [np.zeros((nROC, 4)), np.zeros((nROC, 4))]
    for t in range(min(limit, len(dset))):
        image = dset[t]
        pdt_mask = model.clean(image[0], inpaint=False, binary=False)
        msk = image[2]; use = image[3]
        for i in range(nROC):
            binary_mask = np.array(pdt_mask > threshold[i]) * (1 - use)
            metric[0][i] += maskMetric(binary_mask, msk * (1 - use))
            if dilate is not None:
                binary_mask = dilation(binary_mask, dilate)
                metric[1][i] += maskMetric(binary_mask, msk * (1 - use))
        if (t > limit):
            break
    TP, TN, FP, FN = metric[0][:, 0], metric[0][:, 1], metric[0][:, 2], metric[0][:, 3]
    TP1, TN1, FP1, FN1 = metric[1][:, 0], metric[1][:, 1], metric[1][:, 2], metric[1][:, 3]
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TPR1 = TP1 / (TP1 + FN1)
    FPR1 = FP1 / (FP1 + TN1)

    return (((TPR * 100, FPR * 100), (TPR1 * 100, FPR1 * 100)))


def ROC_LACosmic(dset, sigclip, objlim=2, limit=1e5, dilate=None):
    np.random.seed(0)
    nROC = sigclip.size
    metric = [np.zeros((nROC, 4)), np.zeros((nROC, 4))]
    for t in range(min(limit, len(dset))):
        image = dset[t]
        msk = image[2]
        for i in range(nROC):
            pdt_mask, cleanArr = lac.detect_cosmics(image[0] * 100, sigclip=sigclip[i], sigfrac=0.3, objlim=objlim,
                                                    gain=1, readnoise=5, satlevel=np.inf, sepmed=False,
                                                    cleantype='medmask', niter=4)
            pdt_mask *= (1 - image[3]).astype(bool)
            metric[0][i] += maskMetric(pdt_mask, msk * (1 - image[3]))
            if dilate is not None:
                pdt_mask = dilation(pdt_mask, dilate)
                metric[1][i] += maskMetric(pdt_mask, msk * (1 - image[3]))
    TP, TN, FP, FN = metric[0][:, 0], metric[0][:, 1], metric[0][:, 2], metric[0][:, 3]
    TP1, TN1, FP1, FN1 = metric[1][:, 0], metric[1][:, 1], metric[1][:, 2], metric[1][:, 3]
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TPR1 = TP1 / (TP1 + FN1)
    FPR1 = FP1 / (FP1 + TN1)

    return (((TPR * 100, FPR * 100), (TPR1 * 100, FPR1 * 100)))


def medmask(image, mask):
    clean = np.copy(image)
    xmax = image.shape[0]; ymax = image.shape[1]
    medianImage = np.median(image)
    good = image * (1 - mask)
    pos = np.where(mask)
    for i in range(len(pos[0])):
        x = pos[0 ][i]; y = pos[1][i]
        img = good[max(0, x-2):min(x+3, xmax+1),
                   max(0, y-2):min(y+3, ymax+1)]
        if img.sum()!=0:
            clean[x,y] = np.median(img[img!=0])
        else:
            clean[x,y] = medianImage
    return clean