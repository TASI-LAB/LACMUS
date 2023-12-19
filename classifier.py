import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
import argparse


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    
DEVICE = "cuda:1"
NUM_EPOCHS = 10

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.Y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class MLP(nn.Module):

    def __init__(self):

        # self.__init__()
        super(MLP, self).__init__()
        # Release soon
        
        


    def forward(self, x):
        # Release soon
        return probas

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

def load_model(filename,device):
    model = MLP()
    model.to(device)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model
    
    
def finetune(model, train_loader, device, epochs,learningRate,savepath):

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)  
    for epoch in range(epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            
            features = features.to(device)
            targets = targets.to(device)
            probas = model(features)
            cost = F.cross_entropy(probas, targets)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            print('Epoch: %03d/%03d | Train: %.3f%%' % (
                epoch+1, NUM_EPOCHS, 
                compute_accuracy(model, train_loader, device=device)))
            
    torch.save(model.state_dict(), savepath)
    

def train():
    train_dataset = datasets.MNIST(root='data', 
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    download=True)
    test_dataset = datasets.MNIST(root='data', 
                                train=False, 
                                transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=128, 
                            shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=128, 
                            shuffle=False)
  

    model = MLP()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  
    
    for epoch in range(NUM_EPOCHS):
    
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            probas = model(features)
            cost = F.cross_entropy(probas, targets)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            print('Epoch: %03d/%03d | Train: %.3f%%' % (
                epoch+1, NUM_EPOCHS, 
                compute_accuracy(model, train_loader, device=DEVICE)))
        
    with torch.set_grad_enabled(False): # save memory during inference
        print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE)))
    torch.save(model.state_dict(), 'mnist.pth')
    



if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str, default="VQ-VAE",
                        help='name of the data folder')
    
    train()
    
    
    