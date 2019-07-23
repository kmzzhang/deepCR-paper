from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pickle

class data(Dataset):
    def __init__(self, file=None, part=None, field=[], data=None, random=False, augment=None, random_nmask=True, aug_sky=[0,0]):
        """Initializa the HSTdata class.

        Parameters:
            file: filename including directory
        """
        np.random.seed(1)
        # if type(test) != list:
        #     test = [test]
        self.fsplit = 0.8
        self.part = part
        self.data = data
        self.augment = augment
        self.readPickle(file, field, random)
        self.len = self.raw.shape[0]
        self.aug_sky = aug_sky
        self.random_nmask = random_nmask
        del self.data

    def readPickle(self, file, field, random):
        """Read training set formatted as a dictionary

        Parameters:
            file: filename including directory
        """
        if self.data is None:
            file = open(file, "rb")
            self.data = pickle.load(file)
            file.close()
        # iterate through dictionary to build dataset
        raw = [];
        clean = [];
        crmask = [];
        badmask = [];
        exp = [];
        gain = []
        sky = []
        f = self.fsplit

        for key in self.data.keys():

            if (self.data[key]['type'] not in field and field != []):
                continue

            for i in range(len(self.data[key]['single'])):
                _exp = self.data[key]['exposure'][i]
                _gain = self.data[key]['gain'][i]
                try:
                    _sky = self.data[key]['gain'][i]
                except:
                    _sky = 0
                n_stamp = self.data[key]['stacked'][i].shape[0]

                if (self.part == 'train'):
                    clean.append(self.data[key]['stacked'][i][:int(n_stamp * f)])
                    raw.append(self.data[key]['single'][i][:int(n_stamp * f)])
                    crmask.append(self.data[key]['crmask'][i][:int(n_stamp * f)])
                    badmask.append(self.data[key]['badmask'][i][:int(n_stamp * f)])

                elif (self.part == 'val'):
                    clean.append(self.data[key]['stacked'][i][int(n_stamp * f):])
                    raw.append(self.data[key]['single'][i][int(n_stamp * f):])
                    crmask.append(self.data[key]['crmask'][i][int(n_stamp * f):])
                    badmask.append(self.data[key]['badmask'][i][int(n_stamp * f):])

                else:
                    clean.append(self.data[key]['stacked'][i])
                    raw.append(self.data[key]['single'][i])
                    crmask.append(self.data[key]['crmask'][i])
                    badmask.append(self.data[key]['badmask'][i])

                n = clean[-1].shape[0]
                exp.append([_exp] * n)
                gain.append([_gain] * n)
                sky.append([_sky] * n)

        index = np.arange(np.concatenate(clean).shape[0])
        if random:
            np.random.shuffle(index)
        self.clean = np.concatenate(clean)[index]
        self.raw = np.concatenate(raw)[index]
        self.crmask = np.concatenate(crmask)[index]
        self.badmask = np.concatenate(badmask)[index]
        self.exp = np.concatenate(exp)[index]
        self.gain = np.concatenate(gain)[index]
        self.sky = np.concatenate(sky)[index]

    def __len__(self):
        return self.raw.shape[0]

    def __getitem__(self, i):
        a = self.aug_sky[0] + np.random.rand() * (self.aug_sky[1] - self.aug_sky[0])
        a = a * self.sky[i]

        if self.augment is not None:
            if self.random_nmask:
                size = np.random.randint(1, self.augment)
            else:
                size = self.augment
            index = np.random.randint(self.len, size=size)
            index = np.concatenate([index, np.array([i])])
            mask = self.crmask[index].sum(axis=0) > 0
            badmask = self.badmask[i].astype(bool) + self.crmask[i].astype(bool)
        else:
            mask = self.crmask[i]
            badmask = self.badmask[i]
        return self.raw[i] + a, self.clean[i] + a, mask, badmask, self.exp[i], \
               self.gain[i]