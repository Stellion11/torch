<<<<<<< HEAD
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)

#1. 데이터
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=52,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print("============================================")
print(x_train.dtype)
print(x_train.shape, y_train.shape)
print(type(x_train))
'''
torch.float32
torch.Size([398, 30]) torch.Size([398, 1])
<class 'torch.Tensor'>
'''

#2. 모델구성
model = nn.Sequential(
    nn.Linear(30, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.SiLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
).to(DEVICE)


#3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.04)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    # accuracy 계산
    y_pred_labels = (hypothesis > 0.5).float()
    acc = accuracy_score(y.cpu(), y_pred_labels.cpu())
    return loss.item(), acc

epochs = 500
for epoch in range(1, epochs+1):
    loss, acc = train(model, criterion, optimizer, x_train, y_train)
    print(f'epoch: {epoch:4d}, loss: {loss:.6f}, acc: {acc:.4f}')
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        y_pred_labels = (y_pred > 0.5).float()
        acc = accuracy_score(y.detach().cpu().numpy(), y_pred_labels.detach().cpu().numpy())
        loss_fin = criterion(y_pred, y)
    return loss_fin.item(), acc

loss_fin, acc = evaluate(model, criterion, x_test, y_test)

print('최종 loss:', loss_fin)
print('최종 accuracy:', acc)

# lr = 0.02
# 최종 loss: 0.0416441485285759
=======
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)

#1. 데이터
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=52,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print("============================================")
print(x_train.dtype)
print(x_train.shape, y_train.shape)
print(type(x_train))
'''
torch.float32
torch.Size([398, 30]) torch.Size([398, 1])
<class 'torch.Tensor'>
'''

#2. 모델구성
model = nn.Sequential(
    nn.Linear(30, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.SiLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
).to(DEVICE)


#3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.04)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    # accuracy 계산
    y_pred_labels = (hypothesis > 0.5).float()
    acc = accuracy_score(y.cpu(), y_pred_labels.cpu())
    return loss.item(), acc

epochs = 500
for epoch in range(1, epochs+1):
    loss, acc = train(model, criterion, optimizer, x_train, y_train)
    print(f'epoch: {epoch:4d}, loss: {loss:.6f}, acc: {acc:.4f}')
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        y_pred_labels = (y_pred > 0.5).float()
        acc = accuracy_score(y.detach().cpu().numpy(), y_pred_labels.detach().cpu().numpy())
        loss_fin = criterion(y_pred, y)
    return loss_fin.item(), acc

loss_fin, acc = evaluate(model, criterion, x_test, y_test)

print('최종 loss:', loss_fin)
print('최종 accuracy:', acc)

# lr = 0.02
# 최종 loss: 0.0416441485285759
>>>>>>> 16a711e (initialize torch repo)
# 최종 accuracy: 0.9941520467836257