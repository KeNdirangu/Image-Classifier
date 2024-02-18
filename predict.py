# Adopted from https://github.com/paulstancliffe/Udacity-Image-Classifier/blob/master/predict.py
import argparse
import json
import PIL
from PIL import Image
import torch
from torch import nn, optim
import numpy as np
from math import ceil
from train import check_gpu
from torchvision import models, transforms, datasets

def arg_parser():
    parser = argparse.ArgumentParser(description = "predict.py")
    parser.add_argument('--image', type=str,help='Point to image file for prediction.',required=True)
    parser.add_argument('--checkpoint', type=str,help='Point to checkpoint file as str.',required=True)
    parser.add_argument('--top_k', type=int,help='Choose top K matches as int.')
    parser.add_argument('--category_names', dest='category_names',action='store',default='cat_to_name.json')
    parser.add_argument('--gpu', default='gpu',action='store',dest='gpu')
    
    args = parser.parser_args()
    
    return args

#call variables with default values
arch= ''
checkpoint = 'checkpoint.pth'
image = 'flowers/test/1/image_06752.jpg' #rem to change image
category_names = 'cat_to_name.json'
top_k = 5

#harness commandline arguements
if args.checkpoint:
    checkpoint = args.checkpoint
if args.image:
    image = args.image
if args.category_names:
    category_names = args.category_names
if args.gpu:
    torch.cuda(torch.cuda.is_available())

with open(filepath, 'r') as f:
    category_name = json.load(f)
    
def loading_the_checkpoint(path = 'checkpoint_path'):
    checkpoint = torch.load('checkpoint.pth')
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(chackpoint['state_dict'])
    return model

prac_image = (data_dir + '/test' + '/1/' + 'image_06186.jpg')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Done: Process a PIL image for use in a PyTorch model
    process_img = Image.open(image)
    
    img_transfrom = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    model_image = img_transfrom(process_img)
    
    return image

#adopted from https://www.freecodecamp.org/news/how-to-build-the-best-image-classifier-3c72010b3d55/
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to('cuda')
    image = Image.open(open(image_path))#load image from directory
    image = process_img(image)#process image 
    image = np.expand_dims(image, 0)#flatten images to same 1-dimension vector
    #evaluate image in model and give results
    image = torch.from_numpy(image)
    model.eval()
    with torch.no_grad():
        forward_ps = model(image)[0] #forward pass
        output_ps = probs.exp() #output probabilities
        top_ps, top_cs = probs.topk(topK) #list top categories based on probabilities
        np_ps, np_cs = probs.numpy().astype('float'), cats.numpy() #convert categories and probabilities to numpy 
        ps_cs = {cs_title[str(cat+1)]: prob for prob, cat in zip(probs, cats)}
        
        return ps_cs

model = load_model(checkpoint)
print(model)
        


       