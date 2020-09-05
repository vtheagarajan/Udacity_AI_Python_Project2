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

# Create the parser
my_parser = argparse.ArgumentParser(description='Train a machine learning model to identify images')

# Add the arguments
my_parser.add_argument('--data_dir',
                       type=str,
                       help='the path to list')

my_parser.add_argument('--EarlyExitForDev',type=int,
            help='how many iterations before exiting. 0=no early exit.  to be used during coding, to save time')

my_parser.add_argument('--save_dir',type=str,
                    help='directory to save the model state to')

my_parser.add_argument('--learning_rate',type=float,
                        help='learning rate')

my_parser.add_argument('--epochs',type=int,help='number of epochs for training')

my_parser.add_argument('--check_interval', type=int, help='evaluate training loss and validation accuracy every check_interval batches')

my_parser.add_argument('--train_batch_size',type=int,help='batch size for the training dataset')
my_parser.add_argument('--test_batch_size',type=int,help='batch size for the testing dataset')
my_parser.add_argument('--model_to_use',type=str,help='specify one of two models: VGG-11 or ResNet-18')

# Execute the parse_args() method
args = my_parser.parse_args()

data_dir = args.data_dir
earlyExitForDev = args.EarlyExitForDev
save_dir = args.save_dir
learning_rate = args.learning_rate
epochs = args.epochs
check_interval = args.check_interval
train_batch_size = args.train_batch_size if args.train_batch_size is not None else 40 
test_batch_size = args.test_batch_size if args.test_batch_size is not None else 20
model_to_use = args.model_to_use if args.model_to_use is not None else 'VGG-11'

if learning_rate is None:
    learning_rate = 0.003

if data_dir is not None:
    print(data_dir, os.path.exists(data_dir))
else:
    data_dir = 'C:\\Temp\\DeepLearningImages\\flowers'

if earlyExitForDev is None:
    earlyExitForDev = 1

if epochs is None:
    epochs = 1

if save_dir is None:
    save_dir = os.path.curdir

if check_interval is None:
    check_interval = 1

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


""" if not os.path.isdir(input_path):
    print('The path specified does not exist')
    print(os.path)
    #sys.exit()

print('\n'.join(os.listdir(input_path))) """


train_dir = os.path.join(data_dir,'train')
valid_dir = os.path.join(data_dir,'valid')
test_dir = os.path.join(data_dir,'test')

train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir,transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data,batch_size=train_batch_size,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=test_batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=test_batch_size,shuffle=True)

#print(test_loader.batch_size)

# Build your network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

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





criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device)
print(f"device={device}")
print(model)

#Train the network
checkInterval_loss = 0
train_losses, v_accuracies = [],[]
starttime = time.time()

for epoch in range(epochs):
    for iter_count, (inputs, labels) in enumerate(train_loader):
        #just for development - needs to be removed
        if earlyExitForDev > 0 and iter_count>=earlyExitForDev:
            print (f"Exiting train loop at batch number {iter_count} and epoch {epoch}")
            break
        model.train()
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        log_ps = model.forward(inputs)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        #print(f"loss={loss.item()}")
        checkInterval_loss += loss.item()
        
            
        if iter_count % check_interval == 0:
            #test networkwork accuracy acheived so far
            model.eval()
            sum_accuracy = 0
            
            with torch.no_grad():
                for valid_count, (inputs, labels) in enumerate(valid_loader):
                    #just for development - needs to be removed
                    if earlyExitForDev > 0 and valid_count>=earlyExitForDev:
                        print (f"Exiting validation loop at batch number {valid_count}")
                        break
                    inputs, labels = inputs.to(device), labels.to(device)
                    #Get the log of probabilities
                    log_ps = model.forward(inputs)
                                        
                    #Get the probabilities
                    ps = torch.exp(log_ps)
                    
                    top_p, top_class = torch.topk(ps,1,dim=1)
                    #compare top prediction with label
                    comp_rslt = top_class == labels.view(*top_class.shape)
                    sum_accuracy += torch.mean(comp_rslt.type(torch.FloatTensor)).item()
                    
                    
                
                # calculate values for this check_interval
                validation_accuracy = sum_accuracy/(valid_count + 1)
                train_loss = checkInterval_loss / check_interval
                
                train_losses.append(train_loss)
                v_accuracies.append(validation_accuracy)
                
                checkInterval_loss = 0
                
                print(f"epoch:{epoch},iter_count:{iter_count},train_loss={train_loss:.3f}"
                    f",validation_accuracy={validation_accuracy:.3f}")
            
endtime = time.time()
print(f"Device = {device};Total time for training:{(starttime - endtime):.3f} seconds")

# TODO: Save the checkpoint 
model.class_to_idx = train_data.class_to_idx
check_point = {'model_state_dict' : model.state_dict(),
               'index_vals' : model.class_to_idx,
               'optimizer_state_dict' : optimizer.state_dict(),
               'model_to_use': model_to_use}
curtime = time.localtime()
strcurtime = time.strftime('%Y-%m-%dT%H-%M-%S',curtime)
savepath = os.path.join(save_dir, f"checkpoint_{model_to_use}_{strcurtime}.pth")
torch.save(check_point, savepath)
