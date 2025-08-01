# 토치 11_06 카피

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#####################랜덤고정#####################
import random
SEED = 52
random.seed(SEED)   #파이썬 랜덤 고정
np.random.seed      #넘파이 랜덤 고정
torch.manual_seed(SEED) #토치 랜덤 고정
torch.cuda.manual_seed(SEED) #토치 쿠다 시드 고정!
##################################################

# import warnings
# warnings.filterwarnings('ignore')

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)


#1. 데이터
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=SEED,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print("============================================")
print(x_train.dtype)
print(x_train.shape, y_train.shape)
print(type(x_train))
'''
torch.float32
torch.Size([398, 30]) torch.Size([398, 1])
<class 'torch.Tensor'>
'''

################# torch 데이터셋 만들기 #################
from torch.utils.data import TensorDataset # x, y 합치기
from torch.utils.data import DataLoader    # batch 정의!

###### 1. x와 y를 합칠꺼야
train_set = TensorDataset(x_train, y_train) # tuple 형태로 
test_set = TensorDataset(x_test, y_test)
print(train_set)        #<torch.utils.data.dataset.TensorDataset object at 0x0000018EE4B24940>
print(type(train_set))  #<class 'torch.utils.data.dataset.TensorDataset'>
print(len(train_set))   # 512
print(train_set[0])
print(train_set[0][0])  # 첫번째 x
print(train_set[0][1])  # 첫번째 y

###### 2. batch를 정의한다
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
print(len(train_loader)) # 52
print(train_loader) #<torch.utils.data.dataloader.DataLoader object at 0x000001D86569E070>
# print(train_loader[0]) #에러
# print(train_loader[0][0]) #에러

print('=====================================================')
################# 이터레이터 데이터 확인하기 #################
#1. for 문으로 확인
for aaa in train_loader:
    print(aaa)
    break               # 첫번째 배치 출력

#2. next() 사용
# bbb = iter(train_loader)
# # aaa = bbb.next()        # 파이썬 버전업후 .next()는 없어져서 안써!
# aaa = next(bbb)
# print(aaa)

for x_batch, y_batch in train_loader:
    print(x_batch)
    print(y_batch)
    break


#2. 모델구성
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.silu(x)
        x = self.linear5(x)
        x = self.sig(x)
        return x
    
model = Model(30, 1).to('cuda:0')

#3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.04)

def train(model, criterion, optimizer, loader):
    # model.train()
    total_loss = 0
    
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()
        optimizer.step()
        # total_loss = total_loss + loss.item()
        total_loss += loss.item()
        
        # accuracy 계산
        y_pred_labels = (hypothesis > 0.5).float()
        acc = accuracy_score(y_batch.cpu(), y_pred_labels.cpu())
    return total_loss/len(loader), acc

epochs = 500
for epoch in range(1, epochs+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    print(f'epoch: {epoch:4d}, loss: {loss:.6f}, accuracy:{acc: 4f}')
   
   
#4. 평가, 예측
def evaluate(model, criterion, loader):
    model.eval()
    total_loss=0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_pred = model(x_batch)
            y_pred_labels = (y_pred > 0.5).float()
            acc = accuracy_score(y_pred_labels.detach().cpu().numpy(), y_batch.detach().cpu().numpy())
            loss_fin = criterion(y_pred, y_batch)
            total_loss += loss_fin.item()
    return total_loss/ len(loader), acc

loss_fin, acc = evaluate(model, criterion, test_loader)

print('최종 loss:', loss_fin)
print('최종 accuracy:', acc)

# lr = 0.02
# 최종 loss: 0.0416441485285759
# 최종 accuracy: 0.9941520467836257