<<<<<<< HEAD
# 02, 03, 04, 05
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

#1. 데이터
from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=42,
)

scaler_std = StandardScaler()
x_train = scaler_std.fit_transform(x_train)
x_test = scaler_std.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to('cuda:0')
x_test = torch.tensor(x_test, dtype=torch.float32).to('cuda:0')
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to('cuda:0')
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to('cuda:0')

#2. 모델구성
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
).to('cuda:0')

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    r2 = r2_score(hypothesis.detach().cpu().numpy(), y.detach().cpu().numpy())
    return loss.item(), r2

epochs = 3000
for epoch in range(1, epochs+1):
    loss, r2 = train(model, criterion, optimizer, x_train, y_train)
    print('loss:{}, r2:{}'.format(loss, r2))
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss_fin = criterion(y_pred, y)
        acc_fin = r2_score(y_pred.detach().cpu().numpy(), y.detach().cpu().numpy())
    return loss_fin.item(), acc_fin

loss_fin, acc_fin = evaluate(model, criterion, x_test, y_test)
print('최종 loss:', loss_fin)
print('최종 r2:', acc_fin)
    
# 최종 loss: 0.27523109316825867
=======
# 02, 03, 04, 05
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

#1. 데이터
from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=42,
)

scaler_std = StandardScaler()
x_train = scaler_std.fit_transform(x_train)
x_test = scaler_std.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to('cuda:0')
x_test = torch.tensor(x_test, dtype=torch.float32).to('cuda:0')
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to('cuda:0')
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to('cuda:0')

#2. 모델구성
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
).to('cuda:0')

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    r2 = r2_score(hypothesis.detach().cpu().numpy(), y.detach().cpu().numpy())
    return loss.item(), r2

epochs = 3000
for epoch in range(1, epochs+1):
    loss, r2 = train(model, criterion, optimizer, x_train, y_train)
    print('loss:{}, r2:{}'.format(loss, r2))
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss_fin = criterion(y_pred, y)
        acc_fin = r2_score(y_pred.detach().cpu().numpy(), y.detach().cpu().numpy())
    return loss_fin.item(), acc_fin

loss_fin, acc_fin = evaluate(model, criterion, x_test, y_test)
print('최종 loss:', loss_fin)
print('최종 r2:', acc_fin)
    
# 최종 loss: 0.27523109316825867
>>>>>>> 16a711e (initialize torch repo)
# 최종 r2: 0.7490073442459106