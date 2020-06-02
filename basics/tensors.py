import torch
import numpy as np

# x = torch.tensor([5,3])
# y = torch.tensor([2,1])
# z = torch.zeros([5,4])

# print(x*y)
# print(z)
# print(z.shape)

# x = torch.rand([2,3])
# print(x)
# print(x.view([1,6]))

# x = torch.zeros([2, 4], dtype=torch.int32)
# print(x)
# print(x[0][0])

x = torch.empty(2, 2, 2)
y = torch.rand(2, 2, 2)
z = torch.zeros(2, 2, 2)
k = torch.ones(2, 2, 2, dtype=float)

x = torch.sub(x,y)
print(x)

temp = x[1,1,1]
print(temp.item())

numpy_array = np.array([1,2,3])

torch_tensor = torch.from_numpy(numpy_array)
print(torch_tensor)