#adopted from https://github.com/mishraishan31/Image-Classifier/blob/master/train.py
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchvision import models, datasets, transforms
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from os.path import isdir



def arg_parser():
    parser = argparse.ArgumentParser(description = "Train.py")
    parser.add_argument('data_dir', action = 'store', default='./flower/')
    parser.add_argument('--arch', dest='arch', action='store', default='vgg16', type = str)
    parser.add_argument('--save_dir', dest='save_dir', action='store.', default='./checkpoint.pth')
    parser.add_argument('--learning_rate', dest='learning_rate', action='store.', default = 0.001)
    parser.add_argument('--epochs', type=int, dest='epochs', action='store',default=8)
    parser.add_argument('--gpu', default='gpu',action='store',dest='gpu')
    
    args = parser.parser_args()
    
    return args

#set up command line args
if args.arch:
    arch = args.arch
if args.learning_rate:
    learning_rate = args.learning_rate
if args.gpu:
    torch.cuda(torch.cuda.is_available())
if args.epochs:
    epochs = args.epochs
    
#define transforms
def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=224),
                                     transforms.RandomRotation(30),
                                     transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    Tr_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    return Tr_data

def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    Ts_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    return Ts_data

def validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485,0.456,0.406],0.229,0.224,0.225])
     Vd_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)
#create image and data loaders
img_data_set = [Tr_data, Ts_data, Vd_data] #variable with all datasets 
data_loader_set = [trloader, tsLoader, vdLoader] #variable with all data loaders
                                                
def initial_classifier(model):
    from collections import OrderedDict #to set order in which keys are inserted
    
    classifier = nn.Sequential(OrderedDict([('inputs', nn.Linear(25088, 4096)),
                                       ('relu1', nn.ReLU()),
                                       ('dropout', nn.Dropout(0.05)),
                                       ('fc1', nn.Linear(4096, 1024)),
                                       ('relu2', nn.ReLU()),
                                       ('fc2', nn.Linear(1024, 256)),
                                       ('relu3', nn.ReLU()),
                                       ('fc3', nn.Linear(256, 102)),
                                       ('output', nn.LogSoftmax(dim = 1))]))

    model.classifier = classifier

def network_trainer(model, steps, cuda, criterion, optimizer):
    steps = 0
cuda = torch.cuda.is_available() #use GPU to enhance speed
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)#Adam Optimizer selected to enhance gradient descent

#Prime data for GPU
#Enhance speed using GPU
if cuda:
    model.cuda()
else:
    model.cpu()
    
running_loss = 0
accuracy = 0

#Setting epochs and training
epochs = 8
for epoch in range(epochs):
    #to combine training and validation runs in one cycle increasing code efficiency and training/validation steps
    Tr_run = 0 
    Vd_run = 1
    
    for run in [Tr_run, Vd_run]: #training 
        if run == Tr_run:
            model.train()
        else:
            model.eval() #validation mode
            
        pass_count = 0
     #for loop to load datasets into model primed for GPU usage to enhance speed      
        for data in data_loader_set[run]:
            pass_count += 1
            inputs, labels = data
            if cuda == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                inputs, labels = inputs, labels
    
    
            optimizer.zero_grad() #clear default gradients to prevent accumulation
        
            outputs = model(inputs) #forward pass with model parameters
            loss = criterion(outputs, labels) #calculate loss uing output and labels
            #checking back pass and estimating loss
            if run == Tr_run:
                loss.backward() #backpropagation to adjust weights
                optimizer.step() #update weights
        
            running_loss += loss.item() #calculate the loss 
            probs = torch.exp(outputs).data #calculate probabilities
            equality = (labels.data == probs.max(1)[1])
            accuracy = equality.type_as(torch.cuda.FloatTensor()).mean() #calculate accuracy
        
        if run == Tr_run:
            print('\nEpoch: {}/{}'.format(epoch+1, epochs), '\nTraining Loss: {:.4f} '.format(running_loss/pass_count))
        else:
            print('Validation Loss: {:.4f} '.format(running_loss/pass_count), 'Accuracy: {:.4f} '.format(accuracy))
    
        running_loss = 0
                    
    return model
                                                
trained_model = network_trainer(model, steps, cuda, criterion, optimizer)                                                
def validate_model(model, tsLoader):
    correct,total = 0,0
    with torch.no_grad():
        model.eval()
        for data in tsLoader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('On test images accuracy is: %d%%' % (100 * correct / total))
    
def save_checkpoint(trained_model):
    model.class_to_idx = Tr_data.class_to_idx
    torch.save({'structure' : 'alexnet',
                'fc1': 4096,
                'dropout' : 0.05,
                'epochs' : 8,
                'state_dict' : model.state_dict(),
                'class_to_idx' : model.class_to_idx,
                'optimizer_dict' : optimizer.state_dict()},
                'checkpoint.pth')
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = 'checkpoint.pth'
           
    torch.save(checkpoint, save_dir)
             
            
def main():
    args = arg_paser
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    Tr_data = train_transformer(train_dr)
    Vd_data = train_transformer(valid_dir)
    Ts_data = train_transformer(test_dir)
    
    trloader = data_loader(Tr_data)
    vdloader = data_loader(Vd_data, train=False)
    tsloader = data_loader(Ts_data, train=False)
                                                
    img_data_set = [Tr_data, Ts_data, Vd_data]  
    data_loader_set = [trloader, tsLoader, vdLoader]
    
    model = primaryloader_model(architecture=args.arch)
    model.classifier = initial_classifier(model)
    
    cuda = check_gpu(gpu_arg=args.gpu)
    model.to(cuda);
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print('Learning rate specified as 0.001')
    else: learning_rate = args.learning_rate
            
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    steps = 0
    
    trained_model = network_trainer(model, steps, cuda, criterion, optimizer)
    
    initial_checkpoint(trained_model, args.save_dir,Tr_data)
    
if __name__ == '__main__': main()




    
