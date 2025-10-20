<<<<<<< HEAD
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available() # 대문자면 통상 상수(constant)
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]]).T   
y = np.array([1,2,3,4,5,6,7,8,9,10])   

print(x.dtype) # float64
print(x.shape, y.shape) # (10, 3) (10,)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(x.size(), y.size())
# torch.Size([10, 3]) torch.Size([10, 1])

# 정규화
mean = x.mean()
std = x.std()
x = (x-mean)/std

#2. 모델구성
model = nn.Sequential(
    nn.Linear(3,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1)    
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 100
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))

# 4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss_fin = criterion(y_pred, y)
    return loss_fin.item()

loss_fin = evaluate(model, criterion, x, y)
print('최종 loss_fin:', loss_fin)

x_pred = (torch.tensor([11, 2.0, -1]).to(DEVICE)-mean)/std
result = model(x_pred)
print('[11, 2.0, -1]의 예측값:', result.item())

# lr=0.01
# [11, 2.0, -1]의 예측값: 10.999999046325684
=======
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available() # 대문자면 통상 상수(constant)
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]]).T   
y = np.array([1,2,3,4,5,6,7,8,9,10])   

print(x.dtype) # float64
print(x.shape, y.shape) # (10, 3) (10,)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(x.size(), y.size())
# torch.Size([10, 3]) torch.Size([10, 1])

# 정규화
mean = x.mean()
std = x.std()
x = (x-mean)/std

#2. 모델구성
model = nn.Sequential(
    nn.Linear(3,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1)    
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 100
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))

# 4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss_fin = criterion(y_pred, y)
    return loss_fin.item()

loss_fin = evaluate(model, criterion, x, y)
print('최종 loss_fin:', loss_fin)

x_pred = (torch.tensor([11, 2.0, -1]).to(DEVICE)-mean)/std
result = model(x_pred)
print('[11, 2.0, -1]의 예측값:', result.item())

# lr=0.01
# [11, 2.0, -1]의 예측값: 10.999999046325684
>>>>>>> 16a711e (initialize torch repo)
    