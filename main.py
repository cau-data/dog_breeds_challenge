from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os


# Training settings
parser = argparse.ArgumentParser(description='Dog breed classifier')
parser.add_argument('--model', type=string, default='squeezenet', help='Name or path to the model')
parser.add_argument('--pretrained', type=bool, default=True, help='If the model is pretrained or not, freeze all the layers except the last')
parser.add_argument('--step', type=int, default=4, help='Number of steps before decreasing the learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='Factor by witch you would like to decrease the learning rate')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.08, help='Learning Rate. Default=0.08')
opt = parser.parse_args()

print(opt)

# If there is a gpu return True
use_gpu = torch.cuda.is_available()

# Create a transformed and shuffled training and test dataset of a certain batch size
def get_dataloader(BATCH_SIZE):
    # Transformations for the training and test datasets
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # ImageFolder will give the label corresponding to the folder where the image is and
    # transform it when called. The DataLoader will shuffle and send a batch_size of transformed 
    # images to the choosen model
    train_set = datasets.ImageFolder(root='Images_train/',
                                               transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                                 batch_size=BATCH_SIZE, shuffle=True,
                                                 num_workers=4)

    test_set = datasets.ImageFolder(root='Images_test/',
                                               transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=BATCH_SIZE, shuffle=True,
                                                 num_workers=4)
    # Sets the class names while removing the ten first characters from the folder names
    class_names = train_set.classes
    class_names = [e[10:] for e in class_names]
    
    return train_loader, test_loader, train_set, test_set

# create_model will take in entry the name of the CNN you want to train and if you want it
# pretrained or not. You can load CNN you saved by giving their paths
def create_model(NET, pretrained=True):
    
    if "squeezenet" in NET:        
        model = models.squeezenet1_1(pretrained=pretrained)
        if pretrained == True:
            # If you use a pretrained CNN then we won't change the weights in the layers thus
            # they don't require a gradient
            for param in model.parameters():
                param.requires_grad = False
        # We change the last layer, which is a Conv2d in the case of the squeezenet, so it
        # will return 120 classes instead of 1000
        model.classifier._modules["1"] = nn.Conv2d(512, 120, kernel_size=(1, 1))
        model.num_classes = 120
            
    if "resnet" in NET:
        if "resnet34" in NET:
            model = models.resnet34(pretrained=pretrained)
        else:
            model = models.resnet101(pretrained=pretrained)
        if pretrained == True:
            for param in model.parameters():
                param.requires_grad = False
        # The last layer of the resnet is a linear one, we change it so it returns 120 classes        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 120)    
    # Load the pretrained and saved network
    if "acc" in NET:
        model.load_state_dict(torch.load(NET))
    # If use_gpu = True then we will use cuda to make quick calculations    
    if use_gpu:
        model = model.cuda()
    return model

def train(train_loader, model, criterion, optimizer, scheduler, BATCH_SIZE, EPOCH):
    loss_acc_list = []
    for epoch in range(EPOCH):  # loop over the dataset multiple times

        train_loss = 0
        train_acc = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs) 
            
            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels.data).float().mean()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            train_loss += loss.data[0]
            if i % int(3396/BATCH_SIZE) == int(3396/BATCH_SIZE - 1): # print 5 times per epoch
                train_acc = 100 * train_acc / int(3396/BATCH_SIZE)
                train_loss = train_loss / int(3396/BATCH_SIZE)
                test_loss, test_acc = test_loss_accuracy(test_loader, criterion, model)
                print('[%d, %5d] train loss: %.3f, test loss: %.3f, train accuracy: %.2f %%, test accuracy: %.2f %%' %
                      (epoch + 1, i + 1, train_loss, test_loss, train_acc, test_acc))
                loss_acc_list.append([train_loss, test_loss, train_acc, test_acc])
                train_loss = 0
                train_acc = 0

        scheduler.step()
    return np.array(loss_acc_list)
# Return an array with the loss and accuracy so they can be plotted afterward

# returns the loss and accuracy for the test set
def test_loss_accuracy(loader, criterion, model): 
    correct = 0
    total = 0
    loss = 0
    for data in loader:
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        images.volatile = True
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        correct += (predicted == labels.data).float().mean()       
        loss += criterion(outputs, labels).data[0]
    return loss / total, 100 * correct / total

if __name__ == "__main__":

    train_loader, test_loader, _, _ = get_dataloader(opt.batchsize)
    model = create_model(opt.model, opt.pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter( lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=0.9)
    scheduler= optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)
    loss_acc_list = train(train_loader, model, criterion, optimizer, scheduler, opt.batchsize, opt.epoch)
    #np.save('./logs_your_name', loss_acc_list)
    #torch.save(model.state_dict(), "./your_name")