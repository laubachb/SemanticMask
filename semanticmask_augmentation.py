from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.autograd import Variable

from collections import Counter
import random
import math

def augmentation(x,f_label,f_cluster,number):
    random.shuffle(f_cluster)
    
    f_choose = f_cluster[:number]
    f_remain = f_cluster[number:]
    #print("f_choose:",f_choose)
    #print("f_remain:",f_remain)
    #print(f_label)
    mask_idx=[]
    for i in f_choose: 
        m_idx = np.where(f_label==i)[0]
        p_m = 0.5
        m = np.random.binomial(1, p_m, len(m_idx))>0
        m_idx = m_idx[m] 
        mask_idx.extend(m_idx) 
    #print(mask_idx)
    x_tild = x.clone()
    for i in range(len(x)):
        if i in mask_idx:
            x_tild[i] = 0
    return x_tild,f_remain


    
class MyDataset(Dataset):
    def __init__(self, X, y,f_label,transform=None):
        self.X = torch.from_numpy(X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.from_numpy(y)
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.f_label =f_label
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.X[idx]
        label = self.y[idx]
        f_label = self.f_label 
        f_cluster = list(set(f_label))
        number = len(f_cluster)//2
        x_tild1, f_remain  = augmentation(x,f_label,f_cluster,number)
        x_tild2, f_remain = augmentation(x,f_label,f_remain,number)
        x = [x_tild1, x_tild2]
        return x,label
    
    

def augmentation_position(x,f_label,f_cluster,number):
    random.shuffle(f_cluster)
    f_choose = f_cluster[:number]
    f_remain = f_cluster[number:]
    #print("f_choose:",f_choose)
    #print("f_remain:",f_remain)
    mask_idx=[]
    for i in f_choose: 
        #print(i)
        m_idx = np.where(f_label==i)[0]
        #print(m_idx)
        p_m = 0.4
        m = np.random.binomial(1, p_m, len(m_idx))>0
        m_idx = m_idx[m] #需要mask的index
        mask_idx.extend(m_idx)
       #print(m_idx)
    x_tild = x.clone()
    for i in range(len(x)):
        if i in m_idx:
            x_tild[i] = 0
    m_new = 1 * (x != x_tild)
    return x_tild,f_remain,m_new
    
     
  

def augmentation_description(x,f_label,f_cluster,number):
    random.shuffle(f_cluster)
    f_choose = f_cluster[:number]
    f_remain = f_cluster[number:]
    #print("f_choose:",f_choose)
    #print("f_remain:",f_remain)
    mask_idx=[]
    for i in f_choose: 
        if i == 1:
            p_m = 0.3
        else: 
            p_m = 0.5
        m_idx = np.where(f_label==i)[0]
        m = np.random.binomial(1, p_m, len(m_idx))>0
        mask_idx.extend(m_idx)
    #print(m_idx)
    x_tild = x.clone()
    for i in range(len(x)):
        if i in m_idx:
            x_tild[i] = 0
    #print(x_tild)
    return x_tild,f_remain


    
class MyDataset_position(Dataset):
    def __init__(self, X, y,f_label,transform=None):
        self.X = torch.from_numpy(X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.from_numpy(y)
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.f_label =f_label
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.X[idx]
        label = self.y[idx]
        f_label = self.f_label 
        f_cluster = list(set(f_label))
        number = len(f_cluster)//2
        x_tild1, f_remain,m1 = augmentation_position(x,f_label,f_cluster,number)
        x_tild2, f_remain,m2 = augmentation_position(x,f_label,f_remain,number)
        
        x = [x_tild1, x_tild2]
        m = [m1,m2]
        return x,label,m  
    
    
class MyDataset_description(Dataset):
    def __init__(self, X, y,f_label,transform=None):
        self.X = torch.from_numpy(X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.from_numpy(y)
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.f_label =f_label
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.X[idx]
        label = self.y[idx]
        f_label = self.f_label 
        f_cluster = list(set(f_label))
        number = len(f_cluster)//2
        x_tild1, f_remain  = augmentation(x,f_label,f_cluster,number)
        x_tild2, f_remain = augmentation(x,f_label,f_remain,number)
        x = [x_tild1, x_tild2]
        return x,label 
        
    
class MyDataset_test(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.from_numpy(X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.from_numpy(y)
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.X[idx]
        label = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x,label