import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. 데이터
path = './_data/dacon/따릉이/'          # 시스템 경로에서 시작.

train_csv =  pd.read_csv(path + 'train.csv', index_col=0)     # 0번컬럼을 인덱스컬럼으로 지정 -> 데이터프레임 컬럼에서 제거하고 인덱스로 지정해줌.
print(train_csv)        # [1459 rows x 11 columns] -> [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)  
# test_csv는 predict의 input으로 사용한다.
print(test_csv)         # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission_csv)   # [715 rows x 1 columns]

print(train_csv.shape)      # (1459, 10)
print(test_csv.shape)       # (715, 9)
print(submission_csv.shape) # (715, 1)
# train_csv : 학습데이터
# test_csv : 테스트데이터
# submission_csv : test_csv를 predict하여 예측한 값을 넣어서 제출 

print(train_csv.columns) 
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info())     # non-null수 확인(rows와 비교해서 결측치 수 확인), 데이터 타입 확인

print(train_csv.describe()) # 컬럼별 각종 정보확인할 수 있음 (평균,최댓값, 최솟값 등)

# 1. 데이터

######################################## 결측치 처리 1. 삭제 ########################################
# print(train_csv.isnull().sum())       # 컬럼별 결측치의 갯수 출력
print(train_csv.isna().sum())           # 컬럼별 결측치의 갯수 출력

# train_csv = train_csv.dropna()        # 결측치 제거
# print(train_csv.isna().sum())
# print(train_csv.info())
# print(train_csv)                      # [1328 rows x 10 columns]

######################################## 결측치 처리 2. 평균값 넣기 ########################################
train_csv = train_csv.fillna(train_csv.mean())
print(train_csv.isna().sum())
print(train_csv.info())

########################################  test_csv 결측치 확인 및 처리 ########################################
# test_csv는 결측치 있을 경우 절대 삭제하면 안된다. 답안지에 해당하는(submission_csv)에 채워넣으려면 갯수가 맞아야한다.
print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean())
print('test_csv 정보:', test_csv)
print('ㅡㅡㅡㅡㅡㅡ')

x = train_csv.drop(['count'], axis=1)   # axis = 1 : 컬럼 // axis = 0 : 행
print(x)    # [1459 rows x 9 columns] : count 컬럼을 제거

y = train_csv['count']      # count 컬럼만 추출
print(y.shape)  # (1459,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=42,
)

scaler_std = MinMaxScaler()
x_train = scaler_std.fit_transform(x_train)
x_test = scaler_std.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to('cuda:0')
x_test = torch.tensor(x_test, dtype=torch.float32).to('cuda:0')
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to('cuda:0')
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to('cuda:0')

#2. 모델구성
model = nn.Sequential(
    nn.Linear(9, 128),    
    nn.ReLU(),

    nn.Linear(128, 64),
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
optimizer = optim.Adam(model.parameters(), lr=0.005)

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
        r2_fin = r2_score(y_pred.detach().cpu().numpy(), y.detach().cpu().numpy())
    return loss_fin.item(), r2_fin

loss_fin, r2_fin = evaluate(model, criterion, x_test, y_test)
print('최종 loss:', loss_fin)
print('최종 r2:', r2_fin)

# 최종 loss: 2354.700439453125
# 최종 r2: 0.4395102858543396