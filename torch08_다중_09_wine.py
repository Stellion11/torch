<<<<<<< HEAD
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_wine

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y)) # [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=337,
    stratify=y
)

# 정규화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to('cuda:0')
x_test = torch.tensor(x_test, dtype=torch.float32).to('cuda:0')
y_train = torch.tensor(y_train, dtype=torch.long).to('cuda:0')
y_test = torch.tensor(y_test, dtype=torch.long).to('cuda:0')

#2. 모델구성
model = nn.Sequential(
    nn.Linear(13, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
).to('cuda:0')

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()   # Sparse Categorical Entropy, 즉 one-hot encoding 필요없음
optimizer = optim.Adam(model.parameters(), lr=0.011)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    pred = torch.argmax(hypothesis, dim=1)
    acc = accuracy_score(pred.cpu().numpy(), y.cpu().numpy())
    return loss.item(), acc

epochs = 100
for epoch in range(1, epochs+1):
    loss, acc = train(model, criterion, optimizer, x_train, y_train)
    print('epoch:{}, loss:{}, accuracy:{}'.format(epoch, loss, acc))
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss_fin = criterion(y_pred, y)
        y_pred_arg = torch.argmax(y_pred, dim=1)
        acc_fin = accuracy_score(y_pred_arg.cpu().numpy(), y.cpu().numpy())
    return loss_fin.item(), acc_fin

loss_fin, acc_fin = evaluate(model, criterion, x_test, y_test)
print('최종 loss:', loss_fin)
print('최종 accuracy:', acc_fin)

# 최종 loss: 2.263190981466323e-05
=======
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_wine

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y)) # [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=337,
    stratify=y
)

# 정규화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to('cuda:0')
x_test = torch.tensor(x_test, dtype=torch.float32).to('cuda:0')
y_train = torch.tensor(y_train, dtype=torch.long).to('cuda:0')
y_test = torch.tensor(y_test, dtype=torch.long).to('cuda:0')

#2. 모델구성
model = nn.Sequential(
    nn.Linear(13, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
).to('cuda:0')

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()   # Sparse Categorical Entropy, 즉 one-hot encoding 필요없음
optimizer = optim.Adam(model.parameters(), lr=0.011)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    pred = torch.argmax(hypothesis, dim=1)
    acc = accuracy_score(pred.cpu().numpy(), y.cpu().numpy())
    return loss.item(), acc

epochs = 100
for epoch in range(1, epochs+1):
    loss, acc = train(model, criterion, optimizer, x_train, y_train)
    print('epoch:{}, loss:{}, accuracy:{}'.format(epoch, loss, acc))
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss_fin = criterion(y_pred, y)
        y_pred_arg = torch.argmax(y_pred, dim=1)
        acc_fin = accuracy_score(y_pred_arg.cpu().numpy(), y.cpu().numpy())
    return loss_fin.item(), acc_fin

loss_fin, acc_fin = evaluate(model, criterion, x_test, y_test)
print('최종 loss:', loss_fin)
print('최종 accuracy:', acc_fin)

# 최종 loss: 2.263190981466323e-05
>>>>>>> 16a711e (initialize torch repo)
# 최종 accuracy: 1.0