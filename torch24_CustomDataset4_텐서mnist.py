<<<<<<< HEAD
# mnist로 커스텀데이터셋 만들어봐

# 24-1 카피

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import datasets, transforms 
from torchvision.datasets import MNIST
import torchvision.transforms as tr
from tensorflow.keras.datasets import mnist

#1. 데이터
dataset = load.
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5), (0.5))]) #이미지 증폭(28x28 -> 56x56)
path = './_data/fashionmnist'
# train_dataset = MNIST(path, train=True, download=True, transform=transf)
# test_dataset = MNIST(path, train=False, download=True, transform=transf)
# print(len(train_dataset)) #60000

# x_train, y_train = train_dataset.data/255., train_dataset.targets
# x_test, y_test = test_dataset.data/255., test_dataset.targets
     
#2. 인스턴스 생성
class MyDataset(Dataset):
    def __init__(self, transform=None):         # 전처리 받기
        self.raw_data = MNIST('./_data/fashionmnist', train=True, download=True)
        self.transform = transf                 # 이걸 기억해놔
        
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):                  # 한 장씩 꺼내기
        img, label = self.raw_data[idx]
        if self.transform:                       # 전처리 적용
            img = self.transform(img)
        return img, label

# 3. Dataset 인스턴스화
dataset = MyDataset(transform=transf)

# 4. DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

#4. 출력
for batch_idx, (xb, yb) in enumerate(loader):
    print("================배치: ", batch_idx, '========================')
    print('x: ', xb)
    print('y: ', yb)
=======
# mnist로 커스텀데이터셋 만들어봐

# 24-1 카피

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import datasets, transforms 
from torchvision.datasets import MNIST
import torchvision.transforms as tr
from tensorflow.keras.datasets import mnist

#1. 데이터
dataset = load.
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5), (0.5))]) #이미지 증폭(28x28 -> 56x56)
path = './_data/fashionmnist'
# train_dataset = MNIST(path, train=True, download=True, transform=transf)
# test_dataset = MNIST(path, train=False, download=True, transform=transf)
# print(len(train_dataset)) #60000

# x_train, y_train = train_dataset.data/255., train_dataset.targets
# x_test, y_test = test_dataset.data/255., test_dataset.targets
     
#2. 인스턴스 생성
class MyDataset(Dataset):
    def __init__(self, transform=None):         # 전처리 받기
        self.raw_data = MNIST('./_data/fashionmnist', train=True, download=True)
        self.transform = transf                 # 이걸 기억해놔
        
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):                  # 한 장씩 꺼내기
        img, label = self.raw_data[idx]
        if self.transform:                       # 전처리 적용
            img = self.transform(img)
        return img, label

# 3. Dataset 인스턴스화
dataset = MyDataset(transform=transf)

# 4. DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

#4. 출력
for batch_idx, (xb, yb) in enumerate(loader):
    print("================배치: ", batch_idx, '========================')
    print('x: ', xb)
    print('y: ', yb)
>>>>>>> 16a711e (initialize torch repo)
    break