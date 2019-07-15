import numpy as np

def medmask(image, mask):
    """
    inpaint using 5*5 median filter sampling.
    :param image:
    :param mask:
    :return:
    """
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