<<<<<<< HEAD
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]).T
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 텐서형태로 변환
x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

print(x.size(), y.size())
# torch.Size([10, 2]) torch.Size([10, 1])

# 정규화
min = x.min()
max = x.max()
x = (x-min)/(max-min)

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(2,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1),
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def training(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1000

for epoch in range(1, epochs+1):
    loss = training(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))

# 4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss_fin = criterion(y_pred, y)
    return loss_fin.item()

loss_fin = evaluate(model, criterion, x, y)
print('최종 loss:', loss_fin)

x_pred = (torch.Tensor([[11, 1.2]]).to(DEVICE)-min)/(max-min)
result = model(x_pred)
=======
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]).T
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 텐서형태로 변환
x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

print(x.size(), y.size())
# torch.Size([10, 2]) torch.Size([10, 1])

# 정규화
min = x.min()
max = x.max()
x = (x-min)/(max-min)

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(2,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1),
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def training(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1000

for epoch in range(1, epochs+1):
    loss = training(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))

# 4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss_fin = criterion(y_pred, y)
    return loss_fin.item()

loss_fin = evaluate(model, criterion, x, y)
print('최종 loss:', loss_fin)

x_pred = (torch.Tensor([[11, 1.2]]).to(DEVICE)-min)/(max-min)
result = model(x_pred)
>>>>>>> 16a711e (initialize torch repo)
print('[11, 1.2]의 예측값:', result.item())