import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

raw_training_set = datasets.MNIST("", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
raw_testing_set = datasets.MNIST("", train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))

training_set = torch.utils.data.DataLoader(raw_training_set, batch_size=64, shuffle=True)
testing_set = torch.utils.data.DataLoader(raw_testing_set, batch_size=64, shuffle=True)

for data in training_set:
    x,y = data[0][0].view(28,28), data[1][0]
    plt.imshow(x)
    plt.show()
    print(y)
    break

balance_counter = [0]*10
for data in training_set:
    x_list, y_list = data
    for y in y_list:
        balance_counter[int(y)] += 1

print(balance_counter)
