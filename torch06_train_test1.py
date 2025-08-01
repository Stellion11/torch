import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

# 평가데이터(30%)
x_test = np.array([8,9,10,11])
y_test = np.array([8,9,10,11])

x_pre = np.array([12,13,14])

x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_pre = torch.tensor(x_pre, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# 정규화
mean = x_train.mean()
std = x_train.std()
x_train = (x_train-mean)/std
x_test = (x_test-mean)/std
x_pre = (x_pre-mean)/std

#2. 모델구성
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 1)
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

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch:{}, loss:{}'.format(epoch, loss))
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    with torch.no_grad():
        y_pred = model(x)
        loss_fin = criterion(y_pred, y)
    return loss_fin.item()

loss_fin = evaluate(model, criterion, x_test, y_test)
print('최종 loss: ', loss_fin)

result = model(x_pre)
# print('[12, 13, 14]의 예측값:', result.detach()) #2개 이상 출력할때 grad 빼고 나와
print('[12, 13, 14]의 예측값:', result.detach().cpu().numpy()) 
# [12, 13, 14]의 예측값: [[11.999999]
#  [12.999999]
#  [13.999999]]