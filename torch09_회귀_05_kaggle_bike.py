import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. 데이터
path = './_data/kaggle/bike/'           # 상대경로 : 대소문자 구분X

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

print(train_csv)
print(train_csv.shape)          # (10886, 11)
print(test_csv.shape)           # (6493, 8)
print(submission_csv.shape)     # (6493, 2)

# 결측치 확인
print(train_csv.info())
print(train_csv.isna().sum())       # 결측치 없음
print(train_csv.isnull().sum())     # 결측치 없음


####### x와 y 분리 ######
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)    # test셋에는 없는 casual, registered, y가 될 count는 x에서 제거
print(x)        # [10886 rows x 8 columns] : 데이터프레임(판다스에서의 행렬)
y = train_csv['count']

print(x.shape, y.shape)
# (10886, 8) (10886,)

# 로그 변환
y = np.log1p(train_csv['count'])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to('cuda:0')
x_test = torch.tensor(x_test, dtype=torch.float32).to('cuda:0')
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to('cuda:0')
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to('cuda:0')

#2. 모델구성
model = nn.Sequential(
    nn.Linear(8, 64),    
    nn.ReLU(),

    nn.Linear(64, 32),
    nn.ReLU(),

    nn.Linear(32, 16),
    nn.ReLU(),

    nn.Linear(16, 8),
    nn.ReLU(),

    nn.Linear(8, 1),           
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

epochs = 500
for epoch in range(1, epochs+1):
    loss, r2 = train(model, criterion, optimizer, x_train, y_train)
    print('loss:{}, r2:{}'.format(loss, r2))
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss_fin = criterion(y_pred, y)
        
        # log 스케일 → 원래 스케일로 역변환 후 r2 계산
        y_pred_real = np.expm1(y_pred.cpu().numpy())
        y_real = np.expm1(y.cpu().numpy())
        r2_fin = r2_score(y_real, y_pred_real)
        
    return loss_fin.item(), r2_fin

loss_fin, r2_fin = evaluate(model, criterion, x_test, y_test)
print('최종 loss:', loss_fin)
print('최종 r2:', r2_fin)

# 최종 loss: 2354.700439453125
# 최종 r2: 0.4395102858543396

# 훈련 끝나고 평가 또는 제출할 때 다시 원래대로
pred = np.expm1(model(x_test).cpu().detach().numpy())

# 최종 loss: 1.4584413766860962
# 최종 r2: 0.11303126811981201