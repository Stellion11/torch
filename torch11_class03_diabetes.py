import numpy as np
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

#1. 데이터
from sklearn.datasets import load_diabetes
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=SEED,
)

scaler_std = MinMaxScaler()
x_train = scaler_std.fit_transform(x_train)
x_test = scaler_std.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to('cuda:0')
x_test = torch.tensor(x_test, dtype=torch.float32).to('cuda:0')
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to('cuda:0')
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to('cuda:0')

#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(10, 128),    
#     nn.Dropout(0.3),
#     nn.ReLU(),

#     nn.Linear(128, 64),
#     nn.Dropout(0.3),
#     nn.ReLU(),

#     nn.Linear(64, 32),
#     nn.Dropout(0.3),
#     nn.ReLU(),

#     nn.Linear(32, 16),
#     nn.Dropout(0.3),
#     nn.ReLU(),

#     nn.Linear(16, 8),
#     nn.Dropout(0.3),
#     nn.ReLU(),

#     nn.Linear(8, 1),           
# ).to('cuda:0')

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, 8)
        self.linear6 = nn.Linear(8, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        x = self.linear3(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        x = self.linear4(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        x = self.linear5(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        x = self.linear6(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x
    
model = Model(10, 1).to('cuda:0')

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
        r2_fin = r2_score(y_pred.detach().cpu().numpy(), y.detach().cpu().numpy())
    return loss_fin.item(), r2_fin

loss_fin, r2_fin = evaluate(model, criterion, x_test, y_test)
print('최종 loss:', loss_fin)
print('최종 r2:', r2_fin)

