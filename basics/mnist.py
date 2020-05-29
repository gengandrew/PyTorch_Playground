import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as func
from torchvision import transforms, datasets

epochs = 10

raw_training_set = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
raw_testing_set = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
training_set = torch.utils.data.DataLoader(raw_training_set, batch_size=64, shuffle=True)
testing_set = torch.utils.data.DataLoader(raw_testing_set, batch_size=64, shuffle=True)

# for data in training_set:
#     x,y = data[0][0].view(28,28), data[1][0]
#     plt.imshow(x)
#     plt.show()
#     print(y)
#     break

# balance_counter = [0]*10
# for data in training_set:
#     x_list, y_list = data
#     for y in y_list:
#         balance_counter[int(y)] += 1

# print(balance_counter)

class Neural_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = self.fc4(x)
        return func.log_softmax(x, dim=1)

Net = Neural_Net()
optimizer = optim.Adam(Net.parameters(), lr=0.001)

for epoch in range(epochs):
    for data in training_set:
        x,y = data
        Net.zero_grad()
        output = Net(x.view(-1, 28*28))
        loss = func.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    
    print(loss)

correct = 0
total = 0
with torch.no_grad():
    for data in training_set:
        x,y = data
        output = Net(x.view(-1, 28*28))
        for i in range(0, output.shape[0]):
            predication = torch.argmax(output[i])
            if predication == int(y[i]):
                correct += 1
            
            total += 1

print("Training Set Accuracy: ", correct/total)

correct = 0
total = 0
with torch.no_grad():
    for data in testing_set:
        x,y = data
        output = Net(x.view(-1, 28*28))
        for i in range(0, output.shape[0]):
            predication = torch.argmax(output[i])
            if predication == int(y[i]):
                correct += 1
            
            total += 1

print("Testing Set Accuracy: ", correct/total)
