<<<<<<< HEAD
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

x = np.array([range(10), range(21, 31), range(201, 211)]).T    
y = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]
              ]).T 

print(x.shape, y.shape) # (10, 3) (10, 3)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

# 정규화
mean = x.mean(dim=0, keepdim=True)  # (1, 3)
std = x.std(dim=0, keepdim=True)    # (1, 3)
x = (x-mean)/std

#2. 모델구성
model = nn.Sequential(
    nn.Linear(3, 6),
    nn.ReLU(),
    nn.Linear(6, 5),
    nn.ReLU(),
    nn.Linear(5, 4),
    nn.ReLU(),
    nn.Linear(4, 3)
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch:{}, loss:{}'.format(epoch, loss))
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    with torch.no_grad():
        y_pred = model(x)
        loss_fin = criterion(y_pred, y)
    return loss_fin.item()

loss_fin = evaluate(model, criterion, x, y)
print('최종 loss: ', loss_fin)

x_pred = torch.Tensor([10, 31, 211]).to(DEVICE)
x_pred = (x_pred-mean)/std
result = model(x_pred)
# print('[10, 31, 211]의 예측값:', result.detach()) #2개 이상 출력할때 grad 빼고 나와
print('[10, 31, 211]의 예측값:', result.detach().cpu().numpy()) 

# 원하는 값: [11, 0, -1]
=======
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

x = np.array([range(10), range(21, 31), range(201, 211)]).T    
y = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]
              ]).T 

print(x.shape, y.shape) # (10, 3) (10, 3)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

# 정규화
mean = x.mean(dim=0, keepdim=True)  # (1, 3)
std = x.std(dim=0, keepdim=True)    # (1, 3)
x = (x-mean)/std

#2. 모델구성
model = nn.Sequential(
    nn.Linear(3, 6),
    nn.ReLU(),
    nn.Linear(6, 5),
    nn.ReLU(),
    nn.Linear(5, 4),
    nn.ReLU(),
    nn.Linear(4, 3)
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch:{}, loss:{}'.format(epoch, loss))
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    with torch.no_grad():
        y_pred = model(x)
        loss_fin = criterion(y_pred, y)
    return loss_fin.item()

loss_fin = evaluate(model, criterion, x, y)
print('최종 loss: ', loss_fin)

x_pred = torch.Tensor([10, 31, 211]).to(DEVICE)
x_pred = (x_pred-mean)/std
result = model(x_pred)
# print('[10, 31, 211]의 예측값:', result.detach()) #2개 이상 출력할때 grad 빼고 나와
print('[10, 31, 211]의 예측값:', result.detach().cpu().numpy()) 

# 원하는 값: [11, 0, -1]
>>>>>>> 16a711e (initialize torch repo)
# [10, 31, 211]의 예측값: [[10.421776    0.46191192 -0.2351464 ]]