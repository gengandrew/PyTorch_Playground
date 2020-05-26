import torch

x = torch.tensor([5,3])
y = torch.tensor([2,1])
z = torch.zeros([5,4])

print(x*y)
print(z)
print(z.shape)

x = torch.rand([2,3])
print(x)
print(x.view([1,6]))

x = torch.zeros([2, 4], dtype=torch.int32)
print(x)
print(x[0][0])