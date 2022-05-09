from multiprocessing import freeze_support

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision.transforms import transforms
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import numpy as np
import os
import math

from pathlib import Path
from datetime import datetime

# Hyperparameters.
num_epochs = 30
batch_size = 256  # 128, 256
weight_decay = 0.01
learning_rate = 0.001

num_classes = 6
num_of_workers = 2


DATA_PATH_TRAIN = Path(
    'D:\\Data_mining\\Dataset\\train')
DATA_PATH_VAL = Path('D:\\Data_mining\\Dataset\\val')

trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Flowers dataset.
train_dataset = datasets.ImageFolder(root=DATA_PATH_TRAIN, transform=trans)
val_dataset = datasets.ImageFolder(root=DATA_PATH_VAL, transform=trans)

# Create custom random sampler class to iter over dataloader.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_of_workers)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_of_workers)

number_of_traning_files = len(train_loader.dataset)
number_of_val_files = len(val_loader.dataset)


# CNN we are going to implement.


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3,
                              out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output


class CNNet(nn.Module):
    def __init__(self, num_class):
        super(CNNet, self).__init__()

        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6, self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128, out_features=num_class)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.fc(output)
        return output


# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

# Create model, optimizer and loss function
model = CNNet(num_classes)

# if cuda is available, move the model to the GPU
if cuda_avail:
    model.cuda()

# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=learning_rate,
                 weight_decay=weight_decay)
# optimizer1 = SGD(model.parameters(), lr=learning_rate,
#                  weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()
# Learing rate scheduler
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
train_acc_arr = []
train_loss_arr = []
val_acc_arr = []
val_loss_arr = []


def show_loss_accuracy():

    fig = plt.figure()
    acc_plt = fig.add_subplot(1, 2, 1)
    acc_plt.plot(train_acc_arr, '-o')
    acc_plt.plot(val_acc_arr, '-o')
    acc_plt.set_xlabel('epoch')
    acc_plt.set_ylabel('accuracy')
    acc_plt.legend(['Train', 'Valid'])
    acc_plt.set_title('Train vs Valid Accuracy')

    loss_plt = fig.add_subplot(1, 2, 2)
    loss_plt.plot(train_loss_arr, '-o')
    loss_plt.plot(val_loss_arr, '-o')
    loss_plt.set_xlabel('epoch')
    loss_plt.set_ylabel('loss')
    loss_plt.legend(['Train', 'Valid'])
    loss_plt.set_title('Train vs Valid Loss')

    fig_name = datetime.now().strftime(f'%Y-%m-%dT%H-%M-%S')
    plt.savefig(f"D:\\Data_mining\\fig\\{fig_name}.png")
    plt.show()


def save_models(epoch):
    torch.save(model.state_dict(), f"{epoch}.model")
    print("Checkpoint saved")


def validate():
    model.eval()
    val_acc = 0.0
    val_loss = 0.0
    for i, (images, labels) in enumerate(val_loader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        # Predict classes using images from the validation set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)

        val_acc += torch.sum(prediction == labels.data).float()

        val_step_loss = loss_fn(outputs, labels)
        val_loss += val_step_loss.item() * images.size(0)

    # Compute the average acc and loss over all 10000 val images
    print(f"Val Acc: {val_acc}")
    val_acc = val_acc / number_of_val_files * 100
    val_loss = val_loss / len(val_loader)
    scheduler.step()

    return (val_acc, val_loss)


def train(num_epoch):
    best_acc = 0.0

    for epoch in range(num_epoch):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the val set
            outputs = model(images)
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data).float()

        learning_rate = optimizer.param_groups[0]['lr']
        print(
            f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        # Compute the average acc and loss over all 50000 training images
        train_acc = train_acc / number_of_traning_files * 100
        train_loss = train_loss / len(train_loader)

        # Evaluate on the val set
        val_acc, val_loss = validate()

        # Save the model if the val acc is greater than our current best
        if val_acc > best_acc:
            save_models(epoch)
            best_acc = val_acc

        train_acc_arr.append(train_acc)
        train_loss_arr.append(train_loss)
        val_acc_arr.append(val_acc)
        val_loss_arr.append(val_loss)

        # Print the metrics
        print(
            f"Epoch {epoch + 1}, Train Accuracy: {train_acc} , TrainLoss: {train_loss} , Val Accuracy: {val_acc}, Val Loss: {val_loss}")


if __name__ == '__main__':
    print("Number of files: {}".format(number_of_traning_files))  # 5198
    print(
        f"Train loader length: {len(train_loader.dataset)}, Val loader length: {len(val_loader.dataset)}")
    freeze_support()
    train(num_epochs)
    show_loss_accuracy()
