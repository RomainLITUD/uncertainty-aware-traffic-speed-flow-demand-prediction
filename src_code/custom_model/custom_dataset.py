import numpy as np
import torch
import zarr
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision import transforms
from torch import nn

np.random.seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AMSdataset(Dataset):
    def __init__(self, years, para, stage='train'):
        self.para = para
        self.interval = para['time_invertal']
        self.Tout = para['horizon']
        self.Tin = para['observation']
        self.stage = stage

        X = []
        print('preprocessing data...')
        for year in years:
            dt = zarr.open('./datasets/'+year+'.zarr')
            xf = get_data(dt, self.Tout, self.Tin, self.interval, self.stage)
            X.append(xf)

        X = np.concatenate(X, 0)
        self.X = X
        print('completed')
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        out = torch.Tensor(self.X[index]).float().to(device)
        return out
    
class AMSdataset_distill(Dataset):
    def __init__(self, years, para, indices, stage='train'):
        self.para = para
        self.interval = para['time_invertal']
        self.Tout = para['horizon']
        self.Tin = para['observation']
        self.stage = stage
        self.indices = indices

        X = []
        print('preprocessing data...')
        for year in years:
            dt = zarr.open('./datasets/'+year+'.zarr')
            xf = get_data(dt, self.Tout, self.Tin, self.interval, self.stage)
            X.append(xf)


        X = np.concatenate(X, 0)
        self.X = X
        print('completed')
    def __len__(self):
        return len(self.indices) + self.surplus

    def __getitem__(self, index):
        order = self.indices[index]
        out = torch.Tensor(self.X[order]).float().to(device)
        return out
    
class AMSdataset_filtered(Dataset):
    def __init__(self, para, mode='high'):
        self.para = para
        self.interval = para['time_invertal']
        self.Tout = para['horizon']
        self.Tin = para['observation']
        self.mode = mode

        dt = np.load('./results/order_train.npz', allow_pickle=True)
        train_indices = dt['order']

        Nb = len(train_indices)

        dt1 = np.load('./results/order_2020.npz', allow_pickle=True)
        dt2 = np.load('./results/order_2021.npz', allow_pickle=True)

        if mode == 'high':
            indices_2020 = dt1['perserve']
            indices_2021 = dt2['perserve']
        else:
            indices_2020 = dt1['remove']
            indices_2021 = dt2['remove']


        X = []
        print('preprocessing data...')
        for year in ['2018', '2019']:
            dt = zarr.open('./datasets/'+year+'.zarr')
            xf = get_data(dt, self.Tout, self.Tin, self.interval, 'train')
            X.append(xf)

        X = np.concatenate(X, 0)[train_indices[int(Nb*0.6):]]

        dt = zarr.open('./datasets/2020.zarr')
        x2020 = get_data(dt, self.Tout, self.Tin, self.interval, 'test')
        x2020 = x2020[indices_2020]

        dt = zarr.open('./datasets/2021.zarr')
        x2021 = get_data(dt, self.Tout, self.Tin, self.interval, 'test')
        x2021 = x2021[indices_2021]

        self.X = np.concatenate([X, x2021, x2020], 0)
        print('completed')
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        out = torch.Tensor(self.X[index]).float().to(device)
        return out
    
def get_data(dt, tout, tin, interval, stage):

    x = []
    V_morning = np.transpose(dt.speed_morning, (0,2,1))
    V_evening = np.transpose(dt.speed_evening, (0,2,1))
    Q_morning = np.transpose(dt.flow_morning, (0,2,1))
    Q_evening = np.transpose(dt.flow_evening, (0,2,1))
    if stage != 'test1':
        V_morning[V_morning>130] = 100.
        V_evening[V_evening>130] = 100.

    V_morning = V_morning/130.
    V_evening = V_evening/130.

    if stage != 'test1':
        Q_morning[Q_morning>3000] = 1000.
        Q_evening[Q_evening>3000] = 1000.

    Q_morning = Q_morning/3000.
    Q_evening = Q_evening/3000.

    # K_morning = Q_morning/V_morning
    # K_evening = Q_evening/V_evening

    T = tout + tin
    if stage == 'train':
        for i in range(0, 120-T, interval):
            status = np.stack([V_morning[:-35,i:i+T], Q_morning[:-35,i:i+T]], -1)
            x.append(status)

        for i in range(0, 210-T, interval):
            status = np.stack([V_evening[:-35,i:i+T], Q_evening[:-35,i:i+T]], -1)
            x.append(status)

        x = np.concatenate(x, 0)
        #np.random.shuffle(x)

    if stage == 'validation':
        for d in range(35):
            for i in range(0, 120-T, interval):
                status = np.stack([V_morning[-d-1,i:i+T], Q_morning[-d-1,i:i+T]], -1)
                x.append(status)
                
            for i in range(0, 210-T, interval):
                status = np.stack([V_evening[-d-1,i:i+T], Q_evening[-d-1,i:i+T]], -1)
                x.append(status)

        x = np.array(x)

    if stage == 'test':
        for d in range(len(V_morning)):
            for i in range(0, 120-T, interval):
                status = np.stack([V_morning[d,i:i+T], Q_morning[d,i:i+T]], -1)
                x.append(status)

        for d in range(len(V_evening)):        
            for i in range(0, 210-T, interval):
                status = np.stack([V_evening[d,i:i+T], Q_evening[d,i:i+T]], -1)
                x.append(status)

        x = np.array(x)

    return x

    



