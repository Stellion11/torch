<<<<<<< HEAD
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import random
SEED = 337
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

# 1.데이터
path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# shape 확인
print(train_csv.shape)          # (652, 9)
print(test_csv.shape)           # (116, 8)
print(submission_csv.shape)     # (116, 2)

###### x와 y 분리 ####
x = train_csv.drop(['Outcome'], axis=1) # (652, 9)
y = train_csv['Outcome']                # (652,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=SEED,
    shuffle=True,
)

print(x_train.shape, x_test.shape)  # (456, 8) (196, 8)
print(y_train.shape, y_test.shape)  # (456,) (196,)

# 이미지 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Tensor 형태로 변환
x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(8, 64),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,1),
#     nn.Sigmoid()    
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sig(x)
        return x

model = Model(8, 1).to('cuda:0')
     
#3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    y_pred_labels = (hypothesis>0.5).float()
    acc = accuracy_score(y_pred_labels.cpu(), y.cpu())
    return loss.item(), acc

epochs = 500
for epoch in range(1, epochs+1):
    loss, acc = train(model, criterion, optimizer, x_train, y_train)
    print(f'epoch:{epoch:4d}, loss:{loss:6f}, accuracy:{acc:4f}')
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        y_pred_label = (y_pred>0.5).float()
        acc = accuracy_score(y_pred_label.cpu(), y.cpu())
        
    return loss.item(), acc

loss_fin, acc = evaluate(model, criterion, x_test, y_test)

print('최종 loss:', loss_fin)
print('최종 accuracy:', acc)

# 최종 loss: 0.4676820635795593
# 최종 accuracy: 0.8015267175572519

=======
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import random
SEED = 337
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

# 1.데이터
path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# shape 확인
print(train_csv.shape)          # (652, 9)
print(test_csv.shape)           # (116, 8)
print(submission_csv.shape)     # (116, 2)

###### x와 y 분리 ####
x = train_csv.drop(['Outcome'], axis=1) # (652, 9)
y = train_csv['Outcome']                # (652,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=SEED,
    shuffle=True,
)

print(x_train.shape, x_test.shape)  # (456, 8) (196, 8)
print(y_train.shape, y_test.shape)  # (456,) (196,)

# 이미지 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Tensor 형태로 변환
x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(8, 64),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,1),
#     nn.Sigmoid()    
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sig(x)
        return x

model = Model(8, 1).to('cuda:0')
     
#3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    y_pred_labels = (hypothesis>0.5).float()
    acc = accuracy_score(y_pred_labels.cpu(), y.cpu())
    return loss.item(), acc

epochs = 500
for epoch in range(1, epochs+1):
    loss, acc = train(model, criterion, optimizer, x_train, y_train)
    print(f'epoch:{epoch:4d}, loss:{loss:6f}, accuracy:{acc:4f}')
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        y_pred_label = (y_pred>0.5).float()
        acc = accuracy_score(y_pred_label.cpu(), y.cpu())
        
    return loss.item(), acc

loss_fin, acc = evaluate(model, criterion, x_test, y_test)

print('최종 loss:', loss_fin)
print('최종 accuracy:', acc)

# 최종 loss: 0.4676820635795593
# 최종 accuracy: 0.8015267175572519

>>>>>>> 16a711e (initialize torch repo)
