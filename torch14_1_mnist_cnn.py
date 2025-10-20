<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST 
import torchvision.transforms as tr

#####################랜덤고정#####################
import random
SEED = 52
random.seed(SEED)   #파이썬 랜덤 고정
np.random.seed      #넘파이 랜덤 고정
torch.manual_seed(SEED) #토치 랜덤 고정
torch.cuda.manual_seed(SEED) #토치 쿠다 시드 고정!
##################################################

# import warnings
# warnings.filterwarnings('ignore')

# pytorch에서 shape은 (RGB, 가로, 세로)

#1. 데이터
transf = tr.Compose([tr.Resize(56), tr.ToTensor()]) #이미지 증폭(28x28 -> 56x56)
# to.Tensor = 토치텐서바꾸기 + minmaxScaler()

path = './_data/torch/'
train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)
print(len(train_dataset)) #60000
# print(train_dataset[0][0])
print(train_dataset[0][1]) #5

img_tensor, label = train_dataset[0]
print(label) #5
print(img_tensor.shape) #torch.Size([1, 56, 56])
print(img_tensor.min(), img_tensor.max()) #tensor(0.) tensor(0.9922) #전처리 돼있음 확인

# x_train, y_train = train_dataset.data/255., train_dataset.targets
# x_test, y_test = test_dataset.data/255., test_dataset.targets

# x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 784) #view reshape이랑 똑같지만 더 빠름

# train_set = TensorDataset(x_train, y_train)
# test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(len(train_loader)) # 1875 = 60000/32

#2. 모델구성
class CNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # super(CNN, self).__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1), #(1, 56, 56)->(64,54,54)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2), #(n,64,27,27)
            nn.Dropout(0.2),            
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1), #(n,64,27,27)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), #(n,32,25,25)
            nn.Dropout(0.2),                      
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1), #(n,16,10,10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), #(n,16,5,5)
            nn.Dropout(0.2),  
        )
        
        self.flatten = nn.Flatten()         
        
        self.hidden_layer4 = nn.Sequential( # flatten에서 받아라.
            nn.Linear(16*5*5, 64),
            nn.ReLU(),
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.flatten(x)
        # x = x.view(x.shape[0], -1) # flatten
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x
    
model = CNN(1).to('cuda:0') # torch에서는 channel만 input으로 넣어줌, 나머지는 알아서 맞춰줌   

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

# 최종 loss: 0.09517033430974567
=======
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST 
import torchvision.transforms as tr

#####################랜덤고정#####################
import random
SEED = 52
random.seed(SEED)   #파이썬 랜덤 고정
np.random.seed      #넘파이 랜덤 고정
torch.manual_seed(SEED) #토치 랜덤 고정
torch.cuda.manual_seed(SEED) #토치 쿠다 시드 고정!
##################################################

# import warnings
# warnings.filterwarnings('ignore')

# pytorch에서 shape은 (RGB, 가로, 세로)

#1. 데이터
transf = tr.Compose([tr.Resize(56), tr.ToTensor()]) #이미지 증폭(28x28 -> 56x56)
# to.Tensor = 토치텐서바꾸기 + minmaxScaler()

path = './_data/torch/'
train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)
print(len(train_dataset)) #60000
# print(train_dataset[0][0])
print(train_dataset[0][1]) #5

img_tensor, label = train_dataset[0]
print(label) #5
print(img_tensor.shape) #torch.Size([1, 56, 56])
print(img_tensor.min(), img_tensor.max()) #tensor(0.) tensor(0.9922) #전처리 돼있음 확인

# x_train, y_train = train_dataset.data/255., train_dataset.targets
# x_test, y_test = test_dataset.data/255., test_dataset.targets

# x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 784) #view reshape이랑 똑같지만 더 빠름

# train_set = TensorDataset(x_train, y_train)
# test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(len(train_loader)) # 1875 = 60000/32

#2. 모델구성
class CNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # super(CNN, self).__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1), #(1, 56, 56)->(64,54,54)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2), #(n,64,27,27)
            nn.Dropout(0.2),            
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1), #(n,64,27,27)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), #(n,32,25,25)
            nn.Dropout(0.2),                      
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1), #(n,16,10,10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), #(n,16,5,5)
            nn.Dropout(0.2),  
        )
        
        self.flatten = nn.Flatten()         
        
        self.hidden_layer4 = nn.Sequential( # flatten에서 받아라.
            nn.Linear(16*5*5, 64),
            nn.ReLU(),
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.flatten(x)
        # x = x.view(x.shape[0], -1) # flatten
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x
    
model = CNN(1).to('cuda:0') # torch에서는 channel만 input으로 넣어줌, 나머지는 알아서 맞춰줌   

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

# 최종 loss: 0.09517033430974567
>>>>>>> 16a711e (initialize torch repo)
# 최종 acc: tensor(0.9702, device='cuda:0')