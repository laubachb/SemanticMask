from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
plt.style.use('default')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.autograd import Variable
from torchvision import transforms as T
from matplotlib import rcParams
from sklearn.manifold import TSNE
from collections import  Counter
import random
import math

from sklearn.decomposition import PCA  # 
import pandas as pd
import numpy as np
from scipy.stats import chi2
from matplotlib import patches
import matplotlib.pyplot as plt
from numpy.linalg import inv,eig, det,pinv
import os
import warnings
warnings.filterwarnings('ignore')



def get_features(model, dataloader):
    total = 0
    model.eval()
    
    for batch_idx, (img, label) in enumerate(dataloader):
        img = img.cuda()
        if batch_idx == 0:
        
            f,_= model(img)
            train_feature = f.cpu().detach().numpy()
        else:
            f,_ = model(img)
            train_feature = np.concatenate((train_feature,f.cpu().detach().numpy()))
    return train_feature

def mahalanobis_distance(train_feature,e):
    centerpoint = np.mean(train_feature , axis=0)  # 
    p1 = e
    p2 = centerpoint
   # Covariance matrix
    covariance  = np.cov(train_feature, rowvar=False)
    if det(covariance) != 0:
        # Covariance matrix power of -1
        covariance_pm1 = np.linalg.matrix_power(covariance, -1)
    else:
        covariance_pm1 = np.linalg.pinv(covariance)
    return (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
    
def evaluate_position(net,trainloader,validloader,testloader):
    net.eval()  
    train_feature= get_features(net,trainloader)
    auroc_max = 0
    with torch.no_grad():         
        for batch_idx, (images, label) in enumerate(validloader):
            images = images.cuda()
            if batch_idx == 0:
                embedding,_ = net(images)
                embedding = np.array(embedding.cpu())
            else:
                emb,_= net(images)
                embedding = np.concatenate((embedding,np.array(emb.cpu())))
        #print(embedding.shape)
        distances = []
        for e in embedding:
            distance = mahalanobis_distance(train_feature,e)
            distances.append(distance)
        distances = np.array(distances)
        
        for batch_idx, (images, label) in enumerate(testloader):
            images = images.cuda()
            if batch_idx == 0:
                embedding,_ = net(images)
                embedding = np.array(embedding.cpu())
                labels = np.array(label)
            else:
                emb,_ = net(images)
                embedding = np.concatenate((embedding,np.array(emb.cpu())))
                labels = np.concatenate((labels,label))
        #print(embedding.shape)
        distances_test = []
        for e in embedding:
            distance_test = mahalanobis_distance(train_feature,e)
            distances_test.append(distance_test)
        distances_test = np.array(distances_test)

        for percentile in range(85,86):
            y_true = []
            y_pred = []
            total_correct = 0
            #print("percentile:",percentile)
            cutoff = np.percentile(distances,percentile)
            pred = distances_test > cutoff
            pred = pred.astype(np.int)
            for i in labels:
                y_true.append(i.item())
            for i in pred:
                y_pred.append(i.item())
            pred = torch.tensor(pred)
            
            labels = torch.tensor(labels)
            total_correct += torch.sum(pred == labels).item()

            cm = confusion_matrix(y_true,y_pred)
            print(cm)
            accuracy = total_correct / len(testloader.dataset)
            from sklearn.metrics import roc_auc_score
            AUROC = roc_auc_score(y_true, distances_test)
            #print("accuracy:",accuracy)
            print("AUCROC:",AUROC)
            #from imblearn.metrics import classification_report_imbalanced
            #print(classification_report_imbalanced(y_true, y_pred, target_names=None))
       
        
    return AUROC