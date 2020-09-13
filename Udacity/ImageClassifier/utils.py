import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import time
import json
from PIL import Image
import numpy as np
import seaborn as sb
import argparse
import os
import sys

def create_model(model_to_use):
    # Build your network

    if model_to_use == 'VGG-11':
        model = models.vgg11(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier = nn.Sequential(nn.Linear(25088,1000),
                                    nn.ReLU(),
                                    nn.Dropout(.2),
                                    nn.Linear(1000,408),
                                    nn.ReLU(),
                                    nn.Dropout(.2),
                                    nn.Linear(408,102),
                                    nn.LogSoftmax(dim=1))
    elif model_to_use == 'ResNet-18':
        model = models.resnet18(pretrained=True)
        model.classifier = nn.Sequential(
                                    nn.Linear(1000,408),
                                    nn.ReLU(),
                                    nn.Dropout(.2),
                                    nn.Linear(408,102),
                                    nn.LogSoftmax(dim=1))
    else:
        raise Exception('Unsupported Model. Please use ResNet-18 or VGG-11')

    return model

def load_saved_checkpoint(checkpointpath):
    # if model state is saved with cuda, then running in cpu mode will give errors
    
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
        
    #print(map_location)
    
    checkpoint = torch.load(checkpointpath, map_location=map_location)
    model = create_model(checkpoint['model_to_use'])
    model.load_state_dict(checkpoint['model_state_dict'])

    model.class_to_idx = checkpoint['index_vals']
    return model

def load_saved_checkpoint_for_training(checkpointpath, learning_rate, device):
    
    if device == 'cuda:0':
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
        
    #print(map_location)
    checkpoint = torch.load(checkpointpath, map_location=map_location)
    model = create_model(checkpoint['model_to_use'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['index_vals']

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

