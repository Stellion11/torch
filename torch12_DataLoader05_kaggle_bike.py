import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import random
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

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
    x, y, train_size=0.8, shuffle=True, random_state=SEED,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        return x
    
model = Model(8, 1).to('cuda:0')
        
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

epochs = 500
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
    acc_fin = r2_score(y_true, y_pred)        
    return total_loss/len(loader), acc_fin

loss_fin, acc_fin = evaluate(model, criterion, test_loader)
print('최종 loss:', loss_fin)
print('최종 acc:', acc_fin)

# 최종 loss: 1.3854589324066604
# 최종 acc: 0.32608115673065186