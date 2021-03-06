from __future__ import print_function, division

import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision


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
    try:
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
    except:
        print('Could not load data. Make sure you have downloaded it!')
        sys.exit(1)
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
            
    elif "resnet" in NET:
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
    else:
        print('Could not find "resnet34"/"resnet101" or "squeezenet" in the specified name of the model')
    # Load the pretrained and saved network
    if "trained" in NET:
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
    # Training settings
    parser = argparse.ArgumentParser(description='Dog breed classifier')
    parser.add_argument('--model', type=str, default='squeezenet', help='Name or path to the model')
    parser.add_argument('--pretrained', type=bool, default=True, help='If the model is pretrained or not, freeze all the layers except the last')
    parser.add_argument('--step', type=int, default=4, help='Number of steps before decreasing the learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='Factor by whitch the learning rate is decreased')
    parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.08, help='Learning Rate. Default=0.08')
    parser.add_argument('--save_name', type=str, default='', help='name of the files to write for the logs.')
    opt = parser.parse_args()
    print(opt)

    # If there is a gpu return True
    use_gpu = torch.cuda.is_available()
    print('GPU usage set to', use_gpu)

    print('Loading data...')
    train_loader, test_loader, _, _ = get_dataloader(opt.batchsize)
    print('Done')

    print('Loading model...')
    model = create_model(opt.model, opt.pretrained)
    print('Done')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter( lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=0.9)
    # learning rate monitor
    scheduler= optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)
    print('Starting training...')
    loss_acc_list = train(train_loader, model, criterion, optimizer, scheduler, opt.batchsize, opt.epoch)
    print('Done')

    # If a name was specified
    if opt.save_name != '':
        np.save(f'./logs_{opt.save_name}_{opt.pretrained}', loss_acc_list)
        torch.save(model.state_dict(), f"./{opt.batchsize}_{opt.model}_{opt.save_name}_trained.pth")
