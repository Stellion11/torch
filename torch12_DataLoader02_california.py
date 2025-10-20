<<<<<<< HEAD
# 02, 03, 04, 05
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

import random
SEED = 42 
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#1. 데이터
from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=SEED,
)

scaler_std = StandardScaler()
x_train = scaler_std.fit_transform(x_train)
x_test = scaler_std.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to('cuda:0')
x_test = torch.tensor(x_test, dtype=torch.float32).to('cuda:0')
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to('cuda:0')
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to('cuda:0')

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

#2. 모델구성
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):                #class 안에 함수는 method
        super(Model, self).__init__() ## nn.Module에 있는 Module과 self 다 쓰겠다
        # super().__init__()  # 위랑 동일
        ### 모델에 대한 정의 ###
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        
    def forward(self, x):               #forward는 nn.Module에서 받아온거
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)  
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        return x
    
model = Model(8, 1).to('cuda:0')
    
#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader):
    total_loss = 0
    y_true = []
    y_pred = []
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        y_true.append(y_batch.detach().cpu())
        y_pred.append(hypothesis.detach().cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy() #concat
    y_pred = torch.cat(y_pred, dim=0).numpy()
    r2 = r2_score(y_true, y_pred)   
    return total_loss/len(loader), r2

epochs = 100
for epoch in range(1, epochs+1):
    loss, r2 = train(model, criterion, optimizer, train_loader)
    print('loss:{}, r2:{}'.format(loss, r2))

#4. 평가, 예측
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            hypothesis = model(x_batch)
            loss_fin = criterion(hypothesis, y_batch)
            total_loss += loss_fin.item()
            
            y_true.append(y_batch.detach().cpu())
            y_pred.append(hypothesis.detach().cpu())
            
    y_true = torch.cat(y_true, dim=0).numpy() #concat
    y_pred = torch.cat(y_pred, dim=0).numpy()        
    r2_fin = r2_score(y_true, y_pred)        
    return total_loss/len(loader), r2_fin

loss_fin, r2_fin = evaluate(model, criterion, test_loader)
print('최종 loss:', loss_fin)
print('최종 r2:', r2_fin)
    
# 최종 loss: 0.33971837097682905
=======
# 02, 03, 04, 05
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

import random
SEED = 42 
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#1. 데이터
from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=SEED,
)

scaler_std = StandardScaler()
x_train = scaler_std.fit_transform(x_train)
x_test = scaler_std.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to('cuda:0')
x_test = torch.tensor(x_test, dtype=torch.float32).to('cuda:0')
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to('cuda:0')
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to('cuda:0')

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

#2. 모델구성
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):                #class 안에 함수는 method
        super(Model, self).__init__() ## nn.Module에 있는 Module과 self 다 쓰겠다
        # super().__init__()  # 위랑 동일
        ### 모델에 대한 정의 ###
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        
    def forward(self, x):               #forward는 nn.Module에서 받아온거
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)  
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        return x
    
model = Model(8, 1).to('cuda:0')
    
#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader):
    total_loss = 0
    y_true = []
    y_pred = []
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        y_true.append(y_batch.detach().cpu())
        y_pred.append(hypothesis.detach().cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy() #concat
    y_pred = torch.cat(y_pred, dim=0).numpy()
    r2 = r2_score(y_true, y_pred)   
    return total_loss/len(loader), r2

epochs = 100
for epoch in range(1, epochs+1):
    loss, r2 = train(model, criterion, optimizer, train_loader)
    print('loss:{}, r2:{}'.format(loss, r2))

#4. 평가, 예측
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            hypothesis = model(x_batch)
            loss_fin = criterion(hypothesis, y_batch)
            total_loss += loss_fin.item()
            
            y_true.append(y_batch.detach().cpu())
            y_pred.append(hypothesis.detach().cpu())
            
    y_true = torch.cat(y_true, dim=0).numpy() #concat
    y_pred = torch.cat(y_pred, dim=0).numpy()        
    r2_fin = r2_score(y_true, y_pred)        
    return total_loss/len(loader), r2_fin

loss_fin, r2_fin = evaluate(model, criterion, test_loader)
print('최종 loss:', loss_fin)
print('최종 r2:', r2_fin)
    
# 최종 loss: 0.33971837097682905
>>>>>>> 16a711e (initialize torch repo)
# 최종 r2: 0.740653932094574