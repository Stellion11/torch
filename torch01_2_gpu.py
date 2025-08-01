import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 DEVICE:', DEVICE)
# torch: 2.7.1+cu118 사용 DEVICE: cuda


# tensorflow의 tf와 약간 다른 torch tf를 사용 

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# x = torch.FloatTensor(x)
# print(x) # tensor([1., 2., 3.]) -> torch tf 형태로 바꿔준거임
# print(x.shape) # torch.Size([3])
# print(x.size()) # torch.Size([3])

# 기본적으로 행렬 형태로 연산을 해야된다.
# 따라서 unsqueeze를 통해 reshape``

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
# unsqueeze는 차원을 늘리는건데 (a,b,c) 중 (0) 이면 a 위치에 1을 추가, (1) 이면 b 위치에....
# print(x)
# print(x.shape) # torch.Size([3, 1])
# print(x.size()) # torch.Size([3, 1])
# print(y.size()) # torch.Size([3, 1])

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1)) # 아웃풋->인풋 순서로 씀
# 위를 torch를 바꾸면 아래와 같다
model = nn.Linear(1, 1).to(DEVICE) # 인풋, 아웃풋 순서로 씀        

# 사실 y = wx + b 가 아니고 y = xw + b
# x가 (n,2) 이면 곱 연산이 되려면 w 는 (2, ...)이 되어야 되기 때문

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# torch에서는 아래와 같이
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.1) # Statistic Gradient Descent(경사하강법)

def train(model, criterion, optimizer, x, y):
    # model.train() # [훈련모드], dropout, batchnormalization 자동적용, default여서 명시하지 않아도 됨.
    # 진짜 중요: 기울기와 가중치는 다르다!
    optimizer.zero_grad() # 기울기 초기화
                          # 각 배치마다 기울기를 초기화(0으로)하여, 기울기 누적에 의한 문제 해결
    hypothesis = model(x) # y = xw + b
    loss = criterion(hypothesis, y) # loss = mse() = 시그마(y - hypothesis)^2 / n
    loss.backward() # 기울기(gradient) 값까지만 계산. e.g 기울기 계산: ∂L/∂w = 3
    optimizer.step() # 가중치 갱신 e.g w ← w - lr * 3
    
    return loss.item() #torch 형태에서 수치형태로 바꿔줘야 값을 볼 수 있음

# 위에까지가 1 epoch 혹은 1 batch 가 됨.

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))

print('==============================================')

# 4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval()    # [평가모드] dropout, batchnormalization 쓰지 않겠다
    with torch.no_grad(): # gradient 갱신을 하지 않겠다
        y_pred = model(x)
        loss_fin = criterion(y, y_pred) # loss의 최종값
    return loss_fin.item()

loss_fin = evaluate(model, criterion, x, y)
print('최종 loss: ', loss_fin)

result = model(torch.Tensor([[4]]).to(DEVICE))
print('4의 예측값: ', result.item())