import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as func

class Neural_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = func.max_pool2d(func.relu(self.conv1(x)), (2,2))
        x = func.max_pool2d(func.relu(self.conv2(x)), (2,2))
        x = func.max_pool2d(func.relu(self.conv3(x)), (2,2))

        if self._to_linear == None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return func.softmax(x, dim=1)

epochs = 3
batch_size = 64
testing_set_size = 2500
is_training_set_built = True

def build_training_set():
    new_image_size = 50
    dir = "./kaggle/PetImages/"
    labels = {"Cat": 0, "Dog": 1}
    training_set = []
    cat_count = 0
    dog_count = 0

    for label in labels:
        for file_name in tqdm(os.listdir(dir + label)):
            try:
                path = os.path.join(dir + label, file_name)
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (new_image_size, new_image_size))
                # cv2.imshow(path, image)
                # cv2.waitKey(0)
                training_set.append([np.array(image), np.eye(2)[labels[label]]])

                if label == "Cat":
                    cat_count += 1
                elif label == "Dog":
                    dog_count += 1
            except Exception:
                pass

    np.random.shuffle(training_set)
    np.save("./kaggle/Cat_Dog_Training_Set.npy", training_set)
    print("Total Cat count is: ", cat_count)
    print("Total Dog count is: ", dog_count)
    print("Training set has been built!")
    return training_set

def load_training_set():
    training_set = np.load("./kaggle/Cat_Dog_Training_Set.npy", allow_pickle=True)
    return training_set

raw_training_set = []
if is_training_set_built == False:
    raw_training_set = build_training_set()
else:
    raw_training_set = load_training_set()

Net = Neural_Net()
optimizer = optim.Adam(Net.parameters(), lr=0.001)
loss_func = nn.MSELoss()

x = []
y = []
for each in raw_training_set:
    x.append(each[0])
    y.append(each[1])

x = torch.Tensor(x).view(-1,50,50)
y = torch.Tensor(y)
x = x/255

training_set_x = x[:-testing_set_size]
training_set_y = y[:-testing_set_size]
testing_set_x = x[-testing_set_size:]
testing_set_y = y[-testing_set_size:]

for epoch in range(epochs):
    for i in tqdm(range(0, len(training_set_x), batch_size)):
        batch_x = training_set_x[i:i+batch_size].view(-1, 1, 50, 50)
        batch_y = training_set_y[i:i+batch_size]

        Net.zero_grad()
        output = Net(batch_x)
        loss = loss_func(output, batch_y)
        loss.backward()
        optimizer.step()
        
    print(loss)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(0, len(training_set_x))):
        x = training_set_x[i]
        y = training_set_y[i]
        
        output = Net(x.view(-1, 1, 50, 50))
        prediction = torch.argmax(output)
        if prediction == torch.argmax(y):
            correct += 1
        
        total += 1

print("Training Set Accuracy: ", correct/total)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(0, len(testing_set_x))):
        x = testing_set_x[i]
        y = testing_set_y[i]
        
        output = Net(x.view(-1, 1, 50, 50))
        prediction = torch.argmax(output)
        if prediction == torch.argmax(y):
            correct += 1
        
        total += 1

print("Testing Set Accuracy: ", correct/total)