import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import random
SEED = 337
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# 1. 데이터
from sklearn.datasets import fetch_covtype
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y)) # [1 2 3 4 5 6 7]
y = y-1 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=SEED,
    stratify=y
)

# 정규화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to('cuda:0')
x_test = torch.tensor(x_test, dtype=torch.float32).to('cuda:0')
y_train = torch.tensor(y_train, dtype=torch.long).to('cuda:0')
y_test = torch.tensor(y_test, dtype=torch.long).to('cuda:0')

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
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x

model = Model(54, 7).to('cuda:0')

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()   # Sparse Categorical Entropy, 즉 one-hot encoding 필요없음
optimizer = optim.Adam(model.parameters(), lr=0.02)

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
    y_pred_labels = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_pred_labels, y_true)  
    return total_loss/len(loader), acc

epochs = 10
for epoch in range(1, epochs+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    print('epoch:{}, loss:{}, accuracy:{}'.format(epoch, loss, acc))
    
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
    y_pred_labels = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_pred_labels, y_true)      
    return total_loss/len(loader), acc

loss_fin, acc = evaluate(model, criterion, test_loader)

print('최종 loss:', loss_fin)
print('최종 accuracy:', acc)

# 최종 loss: 0.6353727901469309
# 최종 accuracy: 0.7435264149806804