import os
import numpy as np
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from random import randint
import torch
import yaml
from collections import namedtuple
import cudf
from tqdm import tqdm
import gc
from pathlib import Path

def toT(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    else:
        return torch.tensor(x).float()

def dual_log1p(x):
    x = x.astype('float32')
    sign = np.sign(x)
    y = np.log1p(np.abs(x))
    return sign*y

def dict_to_namedtuple(dic):
    return namedtuple('Config', dic.keys())(**dic)

def load_yaml_to_dict(path):
    with open(path) as f:
        x = yaml.safe_load(f)
    res = {}
    for i in x:
        res[i] = x[i]['value']
    return res
    
def load_yaml(path):
    res = load_yaml_to_dict(path)
    config = dict_to_namedtuple(res)
    print(config)
    return config

def get_cat_cols():
    return ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120',
                'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

def get_cus_count(df):
    if 'cid' not in df.columns:
        df['cid'],_ = df['customer_ID'].factorize()
    dg = df.groupby('cid').agg({'S_2':'count'})
    dg.columns = ['cus_count']
    dg = dg.reset_index()
    dg = dg.sort_values('cid')
    dg['cus_count'] = dg['cus_count'].cumsum()
    return dg

class RnnDataset(Dataset):

    def __init__(self, df, config):
        self.S = config.seq

        dg = get_cus_count(df)
        self.ids = dg.cus_count.values.astype('int32').get()

        df = self._remove_cols(df)
        target_cols = self._get_target_cols(df, config.tcols)
        self.xids,self.yids = self._get_x_y_cols_ids(df, target_cols)
        self._set_y_mask(df)
        df = self._normalize(df)
        self.data = df.to_pandas().values

    def _get_target_cols(self, df, tcols):
        cols = tcols.split(',')
        if len(cols) == 1 and cols[0] == 'all':
            cat = get_cat_cols()
            return [i for i in df.columns if i not in cat]
        return cols

    def __len__(self):
        return self.ids.shape[0]

    def _set_y_mask(self,df):
        self.tcols = df.columns.values[self.yids]
        #print(self.tcols)
        self.y_mask = df[self.tcols].isnull()

    def get_x_y_dims(self):
        return len(self.xids), len(self.yids)

    def _pad(self,x):
        s = self.S
        mask = np.ones(s)
        if s < x.shape[0]:
            return x[-s:],mask
        m,n = x.shape
        tmp = np.zeros((s-m,n))
        mask[:s-m] = 0
        return np.vstack([tmp,x]),mask

        
    def _remove_cols(self,df):
        not_used = [i for i in df.columns if df[i].dtype=='O']+['cid','S_2']
        print("RnnDataset not used columns:")
        print(not_used)
        cat_cols = get_cat_cols()
        return df.drop(not_used+cat_cols, axis=1)

    def _normalize(self, df):
        for col in df.columns:
            df[col] = dual_log1p(((df[col].fillna(0)*100).astype('int16')*0.01).values)
        return df

    def _get_x_y_cols_ids(self, df, target_cols):
        xids,yids = [],[]
        for c,i in enumerate(df.columns.values):
            if i in target_cols:
                yids.append(c)
            else:
                xids.append(c)
        return xids,yids
    
class TrainRnnDataset(RnnDataset):

    def __getitem__(self, idx):
        if idx == 0:
            s = 0
        else:
            s = self.ids[idx-1].item()
        e = self.ids[idx].item()
        data = self.data[s:e]
        y = data[:,self.yids]

        if y.shape[0] > 2:
            x,mask = self._pad(y[:-2])
            y,_ = self._pad(y[1:-1])
        else:
            x,mask = self._pad(y)
            y = x
        return toT(x),toT(y),toT(mask)

class ValidRnnDataset(RnnDataset):

    def __getitem__(self, idx):
        if idx == 0:
            s = 0
        else:
            s = self.ids[idx-1].item()
        e = self.ids[idx].item()
        data = self.data[s:e]
        y = data[:,self.yids]

        if y.shape[0] > 1:
            x,mask = self._pad(y[:-1])
            y,_ = self._pad(y[1:])
        else:
            x,mask = self._pad(y)
            y = x
        return toT(x),toT(y),toT(mask)
    
class TestRnnDataset(RnnDataset):

    def __getitem__(self, idx):
        if idx == 0:
            s = 0
        else:
            s = self.ids[idx-1].item()
        e = self.ids[idx].item()
        data = self.data[s:e]
        y = data[:,self.yids]
        x,_ = self._pad(y)
        return toT(x)
    
class RNN(pl.LightningModule):
    def __init__(self, x_dim, y_dim, config):
        super(RNN, self).__init__()        
        
        self.config = config
        H = config.H1
        self.gru = nn.GRU(input_size=y_dim, hidden_size=H, 
                          batch_first=True,bidirectional=False, 
                          num_layers=config.layers, dropout=config.dropout)
        self.out = nn.Linear(H, y_dim)
    
    def forward(self, x):
        x0 = x
        x,_ = self.gru(x)
        x = F.relu(x)
        x = self.out(x)
        return x + x0

    def _f(self, batch):
        if len(batch) == 3:
            x,y,mask = batch
            return self(x),x,y,mask
        else:
            assert 0
            
    def training_step(self, batch, batch_nb):
        return self._loss(batch, tag='train')

    def validation_step(self, batch, batch_nb):
        return self._loss(batch, tag='valid', naive=True)

    def predict_step(self, batch, batch_nb):
        yp,_,_,_ = self._f(batch)
        return yp

    def _loss(self, batch, tag, naive=False):
        yp,x2,y2,mask = self._f(batch)
        loss = self._compute_loss(yp,y2,mask,tag)
        if naive:
            self._compute_loss(x2,y2,mask,'naive')
        return loss

    def _compute_loss(self,yp,y2,mask,tag):
        loss = ((yp-y2)**2).mean(dim=-1)
        loss = (loss*mask).sum()/mask.sum()
        lossp = F.mse_loss(yp[:,-1,:],y2[:,-1,:])
        self.log(f'{tag}', loss, prog_bar=True)
        self.log(f'{tag}_last', lossp, prog_bar=True)
        return loss

    def configure_optimizers(self):
        config = self.config
        adam = torch.optim.Adam(self.parameters(), lr=config.lr, 
                                weight_decay=config.wd)
        slr = torch.optim.lr_scheduler.CosineAnnealingLR(adam, 
                                                         config.epochs)
        return [adam], [slr]
            
class AutoRegressiveRNN(nn.Module):
    
    def __init__(self, x_dim, y_dim, config):
        super(AutoRegressiveRNN, self).__init__()        
        
        self.config = config
        H = config.H1
        self.gru = nn.GRU(input_size=y_dim, hidden_size=H, 
                          batch_first=True,bidirectional=False, 
                          num_layers=config.layers, dropout=config.dropout)
        self.out = nn.Linear(H, y_dim)

    def f(self, x):
        x0 = x
        x,_ = self.gru(x)
        x = F.relu(x)
        x = self.out(x)
        return x + x0
        
    def forward(self, x):
        yp = torch.zeros(x.size()[0],13,x.size()[-1]).float().to(x.device)
        for i in range(13):
            p = self.f(x)
            yp[:,i,:] = p[:,-1,:]
            x = torch.cat([x[:,1:,:],p[:,-1:,:]],dim=1)
        return yp