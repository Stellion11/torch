import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import random
SEED = 1799
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

# 1.데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

#  shape 확인
print(train_csv.shape)          # (165034, 13)
print(test_csv.shape)           # (110023, 12)
print(submission_csv.shape)     # (110023, 2)

# 문자 데이터 수치화(인코딩)
from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder() # 인스턴스화
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])   # fit 함수 + transform 함친 합친 함수 : 변환해서 적용
# # 아래 2줄이랑 같다.
# le_geo.fit(train_csv['Geography'])                                    # 'Geography' 컬럼을 기준으로 인코딩한다.
# train_csv['Geography'] = le_geo.transform(train_csv['Geography'])     # 적용하고 train_csv['컬럼']에 입력함.
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])

# 테스트 데이터도 수치화해야한다. 위에서 인스턴스가 이미 fit해놨기때문에 transform만 적용한다.
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])


train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)
print(test_csv.shape)   # (110023, 10)

x = train_csv.drop(['Exited'], axis=1)  
print(x.shape)  # (165034, 10)
y = train_csv['Exited']
print(y.shape)  # (165034,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=SEED,
    shuffle=True,
)

print(x_train.shape, x_test.shape)  # (132027, 10) (33007, 10)
print(y_train.shape, y_test.shape)  # (132027,) (33007,)

# 이미지 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Tensor 형태로 변환
x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(10, 64),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,1),
#     nn.Sigmoid()    
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.sig(x)
        return x
    
model = Model(10, 1).to('cuda:0')
           
#3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    y_pred_labels = (hypothesis>0.5).float()
    acc = accuracy_score(y_pred_labels.cpu(), y.cpu())
    return loss.item(), acc

epochs = 500
for epoch in range(1, epochs+1):
    loss, acc = train(model, criterion, optimizer, x_train, y_train)
    print(f'epoch:{epoch:4d}, loss:{loss:6f}, accuracy:{acc:4f}')
    
#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        y_pred_label = (y_pred>0.5).float()
        acc = accuracy_score(y_pred_label.cpu(), y.cpu())
        
    return loss.item(), acc

loss_fin, acc = evaluate(model, criterion, x_test, y_test)

print('최종 loss:', loss_fin)
print('최종 accuracy:', acc)

# 최종 loss: 0.3286258280277252
# 최종 accuracy: 0.8643318084042779

