import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as func
from torchvision import transforms, datasets

##
# Neural Network Class
##
class Neural_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 196)
        self.fc2 = nn.Linear(196, 98)
        self.fc3 = nn.Linear(98, 49)

        # Bottleneck layers
        self.fc4 = nn.Linear(49, 10)
        self.fc5 = nn.Linear(10, 49)

        self.fc6 = nn.Linear(49, 98)
        self.fc7 = nn.Linear(98, 196)
        self.fc8 = nn.Linear(196, 28*28)
    
    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = func.relu(self.fc4(x))
        x = func.relu(self.fc5(x))
        x = func.relu(self.fc6(x))
        x = func.relu(self.fc7(x))
        x = func.log_softmax(self.fc8(x), dim=1)
        return x

# Parameters for the learning model
epochs = 5
batch_size = 64
is_training_set_built = False

# Obtaining the training/testing sets
raw_training_set = datasets.MNIST("", train=True, download=is_training_set_built, transform=transforms.Compose([transforms.ToTensor()]))
raw_testing_set = datasets.MNIST("", train=False, download=is_training_set_built, transform=transforms.Compose([transforms.ToTensor()]))
training_set = torch.utils.data.DataLoader(raw_training_set, batch_size=batch_size, shuffle=True)
testing_set = torch.utils.data.DataLoader(raw_testing_set, batch_size=batch_size, shuffle=True)

# Determining whether to run the model on the cpu or gpu 
Net = Neural_Net()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Currently running on gpu")
else:
    device = torch.device("cpu")
    print("Currently running on cpu")

# Initializing the model
Net = Neural_Net().to(device)
optimizer = optim.Adam(Net.parameters(), lr=0.001)
loss_func = nn.MSELoss()

# Training the Model
for epoch in range(epochs):
    for data in tqdm(training_set):
        batch_x, batch_y = data
        batch_x = batch_x.view(-1, 28*28)
        Net.zero_grad()
        output = Net(batch_x)
        loss = loss_func(output, batch_x)
        loss.backward()
        optimizer.step()
    
    print(loss)

# Testing the model
with torch.no_grad():
    index = 0
    for data in testing_set:
        batch_x, batch_y = data
        output = Net(batch_x.view(-1, 28*28))
        index = 0

        for i in range(0, output.shape[0]):
            image_out = output[i]
            image_x = batch_x[i]
            
            if index == 4:
                plt.figure()
                ax1 = plt.subplot(1,2,1)
                im1 = ax1.imshow(image_x[0])
                ax1.title.set_text("MNIST Image")
                ax2 = plt.subplot(1,2,2)
                im1 = ax2.imshow(image_out.view(1,28,28)[0])
                ax2.title.set_text("Autoencoder Image")
                plt.show(block=True)
                break
            else:
                plt.figure()
                ax1 = plt.subplot(1,2,1)
                im1 = ax1.imshow(image_x[0])
                ax1.title.set_text("MNIST Image")
                ax2 = plt.subplot(1,2,2)
                im1 = ax2.imshow(image_out.view(1,28,28)[0])
                ax2.title.set_text("Autoencoder Image")
                plt.show(block=False)
        
            index += 1
        
        break