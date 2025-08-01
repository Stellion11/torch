import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

#1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))
x_pred = np.array([101, 102])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=52
)

x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_pred = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# 정규화
x_mean  = x_train.mean()
x_std  = x_train.std()
x_train = (x_train-x_mean)/x_std
x_test = (x_test-x_mean)/x_std
x_pred = (x_pred-x_mean)/x_std

y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train-y_mean)/y_std
y_test = (y_test-y_mean)/y_std


#2. 모델구성
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.Linear(16, 8),
    nn.Linear(8, 4),
    nn.Linear(4, 1)
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

x_pred = model(x_pred)
with torch.no_grad():
    result = model(x_pred)        # 정규화된 y_hat
    result_origin = result * y_std + y_mean  # 역정규화
    
# print('[101, 102]의 예측값:', result.detach()) #2개 이상 출력할때 grad 빼고 나와
print('[101, 102]의 예측값:', result_origin.detach().cpu().numpy()) 
# [101, 102]의 예측값: [[101.999985]
#  [102.99998 ]]