from contrastive_loss import SupConLoss
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
from collections import Counter
import random
import math

from sklearn.decomposition import PCA  
import pandas as pd
import numpy as np
from scipy.stats import chi2
from matplotlib import patches
import matplotlib.pyplot as plt
from numpy.linalg import inv, eig, det, pinv
import os
import warnings
warnings.filterwarnings('ignore')


class ContrastiveEncoder(nn.Module):
    def __init__(self):
        super(ContrastiveEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),              
            )

    def forward(self, x):
        x = self.encoder(x)
        return x

def train_dnn(net, temperature, epochs, optimizer, trainloader_SCL):
    net.train()
    criterion = SupConLoss(temperature)
    training_loss = []
    for epoch in tqdm(range(epochs)):
        runningloss = 0
        for images, labels in trainloader_SCL:
            # Make sure to use CPU instead of CUDA
            images = torch.cat([images[0], images[1]], dim=0).to('cpu')  # Use .to('cpu') instead of .cuda()
            bsz = labels.shape[0]
            features = net.encoder(images)
            f1_1, f2_1 = torch.split(features, [bsz, bsz], dim=0)
            features_1 = torch.cat([f1_1.unsqueeze(1), f2_1.unsqueeze(1)], dim=1)
            features_1 = F.normalize(features_1, dim=2)
            loss = criterion(features_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runningloss += loss.item() / images.shape[0]
        training_loss.append(runningloss)
    
    return net, training_loss
