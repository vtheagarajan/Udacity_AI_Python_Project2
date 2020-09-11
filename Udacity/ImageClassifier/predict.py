import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,models,transforms
#import matplotlib.pyplot as plt
import time
import json
from PIL import Image
import numpy as np
#import seaborn as sb
import argparse
import os
import sys
import utils

# Create the parser
my_parser = argparse.ArgumentParser(description='Train a machine learning model to identify images')

# Add the arguments
my_parser.add_argument('path_to_image',type=str,help='full path to image to predict')
my_parser.add_argument('-chk','--checkpoint_file_with_path',
                       type=str,
                       help='the path to saved model checkpoint file')



# Execute the parse_args() method
args = my_parser.parse_args()

checkpoint_file_with_path = args.checkpoint_file_with_path if args.checkpoint_file_with_path is not None else 'c:\\temp\\checkpoint_VGG-11.pth'
path_to_image = args.path_to_image 

#hard code image name for testing since it does not seem to work from command prompt
#path_to_image = '/home/workspace/ImageClassifier/image_07090.jpg'

if not os.path.isfile(path_to_image):
    print('The path specified does not exist')
    print(path_to_image)
    sys.exit()


# Build your network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model  = utils.load_saved_checkpoint(checkpoint_file_with_path)

print('loaded the model')
# criterion = nn.NLLLoss()
# optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device)
print(f"device={device}")
print(model)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image_path)
    im.thumbnail((256,256))
    
    width, height = im.size   # Get dimensions
    new_width = 224 
    new_height = 224
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
   
    np_image = np.array(im)
    #normalize the image to be 0 To 1
    np_image = np_image/255
    im_mean = np.array([0.485, 0.456, 0.406])
    im_std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - im_mean) / im_std
    np_image = np_image.transpose((2,0,1))
    #print(type(np_image))
    
    #imgtensor = torch.tensor(np_image)
    #imgtensor = imgtensor.float()
    #imgtensor.to(device)
    if device == 'cpu':
        im = torch.from_numpy(np_image).type(torch.FloatTensor)
    else:
        im = torch.from_numpy(np_image).type(torch.cuda.FloatTensor)
    
    #Model expects the first dimension to be batch size, so add batch size of 1
    im = im.unsqueeze(0)
        
    return im



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    im = process_image(image_path)
    #Model expects the first dimension to be batch size, so add batch size of 1
    #im = im.unsqueeze_(0)
    
    
    im.to(device)
    model.eval()
    
    with torch.no_grad():
        
        #Get the log of probabilities
        log_ps = model.forward(im)

        #Get the probabilities
        ps = torch.exp(log_ps)

        #print(f"ps={ps} and label={labels}")

        top_p, top_class = torch.topk(ps,topk,dim=1)
        
        return (top_p,top_class)


probs, classes = predict(path_to_image, model)

# im = process_image(image_path)
#imshow(im,ax1)
probs = probs.to('cpu')
classes = classes.to('cpu')

probs = probs.data.numpy().squeeze()
classes = classes.data.numpy().squeeze()
idx_to_class = {value:key for key, value in model.class_to_idx.items()}
cat_list = [idx_to_class[i] for i in classes if i in idx_to_class] 
print(f"cat_list={cat_list}")
catnames = [cat_to_name[i] for i in cat_list if i in cat_to_name]
#print(f"Expected Flower Name = {cat_to_name[flower_cat]}")
# ax2.barh(catnames,probs)
print(f"Probabilities = {probs}")
print(f"Classes = {catnames}")
dictresults = dict(zip(catnames,probs))
print(dictresults)


