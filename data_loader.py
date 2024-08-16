import scipy.io
import numpy as np
import pandas as pd
import torchvision.datasets as dset
import os
from collections import Counter
from sklearn import preprocessing


def load_saheart():
    with open("saheart.txt") as f:
        data = f.readlines()[14:]
    X = [] 
    y = []
    for line in data:
        if 'Present'in line:
            line = line.replace("Present","1")
        if 'Absent'in line:
            line = line.replace("Absent","0")
        line = line.strip('\n')
        line = line.split(',')
        X.append(list(map(float, line[:-1]) ))
        y.append(float(line[-1]))
    samples = np.array(X)
    labels = np.array(y)

    print("The shape of data:",samples.shape)

    norm_samples = samples[labels == 0]  
    anom_samples = samples[labels == 1]  
    print("The shape of normal data:",norm_samples.shape)
    print("The shape of anomalous data:",anom_samples.shape)

    idx = np.random.permutation(norm_samples.shape[0])   #different permutation will cause different results
    norm_samples = norm_samples[idx]
    n_train = len(norm_samples) // 2  #  
    x_train = norm_samples[:n_train]    
    in_real = norm_samples[n_train:]   

    n_valid = len(in_real) // 2   
    in_valid = in_real[:n_valid]   
    in_test = in_real[n_valid:]    

    x_in = np.concatenate([x_train,in_valid])  
    scaler = preprocessing.MinMaxScaler().fit(x_in)
    x_in = scaler.transform(x_in)  

    x_test = np.concatenate([in_test, anom_samples])

    x_test = scaler.transform(x_test)   

    x_train = x_in[:len(x_train)]  
    y_train=np.zeros(len(x_train))

    x_valid = x_in[len(x_train):] 
    y_valid=np.zeros(len(x_valid))


    y_test = np.concatenate([np.zeros(len(in_test)), np.ones(len(anom_samples))]) 
    return x_train,y_train,x_valid,y_valid,x_test,y_test