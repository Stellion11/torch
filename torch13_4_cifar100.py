<<<<<<< HEAD
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100
import torchvision.transforms as tr

# 데이터 다운
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR100(
    root='./_data/cifar100',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR100(
    root='./_data/cifar100',
    train=False,
    download=True,
    transform=transform
)

#1. 데이터
path = './_data/cifar100'
train_dataset = CIFAR100(path, train=True, download=True)
test_dataset = CIFAR100(path, train=False, download=True)

x_train, y_train = train_dataset.data/255., train_dataset.targets
x_test, y_test = test_dataset.data/255., test_dataset.targets

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

print(x_train.shape, x_test.shape)
# (50000, 32, 32, 3) (10000, 32, 32, 3)

x_train, x_test = x_train.reshape(-1, 32*32*3), x_test.reshape(-1, 32*32*3) #view reshape이랑 똑같지만 더 빠름

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print(x_train.shape, x_test.size())
print(torch.unique(y_train))
# torch.Size([60000, 784]) torch.Size([10000, 784])
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#2. 모델구성
class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # super(DNN, self).__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(32, 100)
    
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x
    
model = DNN(32*32*3).to('cuda:0')    

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1e-4) # 0.0001

def train(model, criterion, optimizer, loader):
    # model.train
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to('cuda:0'), y_batch.to('cuda:0')
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
       
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        
        epoch_loss += loss.item()
        epoch_acc += acc
    
    return epoch_loss / len(loader), epoch_acc/len(loader)

def evaluate(model, criterion, loader):
    loss_fin = 0
    acc_fin = 0
    model.eval()
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to('cuda:0'), y_batch.to('cuda:0')
            
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            y_predict = torch.argmax(y_pred, 1)
            acc = (y_predict == y_batch).float().mean()
            
            loss_fin += loss.item()
            acc_fin += acc
        
        return loss_fin/len(loader), acc_fin/len(loader)

EPOCH=30
for epoch in range(1, EPOCH+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    print(f'epoch:{epoch}, loss:{loss:.4f}, acc:{acc:3f}, \
        val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}'
        )

#4. 평가, 예측    
loss_fin, acc_fin = evaluate(model, criterion, test_loader)
print("===========================================================")
print('최종 loss:', loss_fin)
print('최종 acc:', acc_fin)

=======
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100
import torchvision.transforms as tr

# 데이터 다운
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR100(
    root='./_data/cifar100',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR100(
    root='./_data/cifar100',
    train=False,
    download=True,
    transform=transform
)

#1. 데이터
path = './_data/cifar100'
train_dataset = CIFAR100(path, train=True, download=True)
test_dataset = CIFAR100(path, train=False, download=True)

x_train, y_train = train_dataset.data/255., train_dataset.targets
x_test, y_test = test_dataset.data/255., test_dataset.targets

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

print(x_train.shape, x_test.shape)
# (50000, 32, 32, 3) (10000, 32, 32, 3)

x_train, x_test = x_train.reshape(-1, 32*32*3), x_test.reshape(-1, 32*32*3) #view reshape이랑 똑같지만 더 빠름

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print(x_train.shape, x_test.size())
print(torch.unique(y_train))
# torch.Size([60000, 784]) torch.Size([10000, 784])
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#2. 모델구성
class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # super(DNN, self).__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(32, 100)
    
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x
    
model = DNN(32*32*3).to('cuda:0')    

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1e-4) # 0.0001

def train(model, criterion, optimizer, loader):
    # model.train
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to('cuda:0'), y_batch.to('cuda:0')
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
       
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        
        epoch_loss += loss.item()
        epoch_acc += acc
    
    return epoch_loss / len(loader), epoch_acc/len(loader)

def evaluate(model, criterion, loader):
    loss_fin = 0
    acc_fin = 0
    model.eval()
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to('cuda:0'), y_batch.to('cuda:0')
            
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            y_predict = torch.argmax(y_pred, 1)
            acc = (y_predict == y_batch).float().mean()
            
            loss_fin += loss.item()
            acc_fin += acc
        
        return loss_fin/len(loader), acc_fin/len(loader)

EPOCH=30
for epoch in range(1, EPOCH+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    print(f'epoch:{epoch}, loss:{loss:.4f}, acc:{acc:3f}, \
        val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}'
        )

#4. 평가, 예측    
loss_fin, acc_fin = evaluate(model, criterion, test_loader)
print("===========================================================")
print('최종 loss:', loss_fin)
print('최종 acc:', acc_fin)

>>>>>>> 16a711e (initialize torch repo)
