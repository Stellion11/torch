<<<<<<< HEAD
# 02, 03, 04, 05
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

import random
SEED = 42 
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

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

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

#2. 모델구성
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):                #class 안에 함수는 method
        super(Model, self).__init__() ## nn.Module에 있는 Module과 self 다 쓰겠다
        # super().__init__()  # 위랑 동일
        ### 모델에 대한 정의 ###
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        
    def forward(self, x):               #forward는 nn.Module에서 받아온거
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)  
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        return x
    
model = Model(9, 1).to('cuda:0')
    
#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader):
    total_loss = 0
    y_true = []
    y_pred = []
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        y_true.append(y_batch.detach().cpu())
        y_pred.append(hypothesis.detach().cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy() #concat
    y_pred = torch.cat(y_pred, dim=0).numpy()
    r2 = r2_score(y_true, y_pred)   
    return total_loss/len(loader), r2

epochs = 100
for epoch in range(1, epochs+1):
    loss, r2 = train(model, criterion, optimizer, train_loader)
    print('loss:{}, r2:{}'.format(loss, r2))

#4. 평가, 예측
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            hypothesis = model(x_batch)
            loss_fin = criterion(hypothesis, y_batch)
            total_loss += loss_fin.item()
            
            y_true.append(y_batch.detach().cpu())
            y_pred.append(hypothesis.detach().cpu())
            
    y_true = torch.cat(y_true, dim=0).numpy() #concat
    y_pred = torch.cat(y_pred, dim=0).numpy()
    r2_fin = r2_score(y_true, y_pred)        
    return total_loss/len(loader), r2_fin

loss_fin, r2_fin = evaluate(model, criterion, test_loader)
print('최종 loss:', loss_fin)
print('최종 r2:', r2_fin)
    
# 최종 loss: 2416.044424874442
=======
# 02, 03, 04, 05
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

import random
SEED = 42 
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

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

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

#2. 모델구성
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):                #class 안에 함수는 method
        super(Model, self).__init__() ## nn.Module에 있는 Module과 self 다 쓰겠다
        # super().__init__()  # 위랑 동일
        ### 모델에 대한 정의 ###
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        
    def forward(self, x):               #forward는 nn.Module에서 받아온거
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)  
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        return x
    
model = Model(9, 1).to('cuda:0')
    
#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader):
    total_loss = 0
    y_true = []
    y_pred = []
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        y_true.append(y_batch.detach().cpu())
        y_pred.append(hypothesis.detach().cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy() #concat
    y_pred = torch.cat(y_pred, dim=0).numpy()
    r2 = r2_score(y_true, y_pred)   
    return total_loss/len(loader), r2

epochs = 100
for epoch in range(1, epochs+1):
    loss, r2 = train(model, criterion, optimizer, train_loader)
    print('loss:{}, r2:{}'.format(loss, r2))

#4. 평가, 예측
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            hypothesis = model(x_batch)
            loss_fin = criterion(hypothesis, y_batch)
            total_loss += loss_fin.item()
            
            y_true.append(y_batch.detach().cpu())
            y_pred.append(hypothesis.detach().cpu())
            
    y_true = torch.cat(y_true, dim=0).numpy() #concat
    y_pred = torch.cat(y_pred, dim=0).numpy()
    r2_fin = r2_score(y_true, y_pred)        
    return total_loss/len(loader), r2_fin

loss_fin, r2_fin = evaluate(model, criterion, test_loader)
print('최종 loss:', loss_fin)
print('최종 r2:', r2_fin)
    
# 최종 loss: 2416.044424874442
>>>>>>> 16a711e (initialize torch repo)
# 최종 r2: 0.6493493318557739