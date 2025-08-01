import torch 

x = torch.tensor([
    [[1,2], [3,4], [5,6]],
    [[7,8], [9,10], [11,12]]
])

print(x.shape) #torch.Size([2, 3, 2])

x = x[:, -1, :]

print(x)
# tensor([[ 5,  6],
#         [11, 12]])
print(x.shape) #torch.Size([2, 2])