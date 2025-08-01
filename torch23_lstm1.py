# 22 카피
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import datasets, transforms 
from torchvision.datasets import MNIST
import torchvision.transforms as tr
from sklearn.metrics import accuracy_score, r2_score


#####################랜덤고정#####################
import random
SEED = 52
random.seed(SEED)   #파이썬 랜덤 고정
np.random.seed(SEED)      #넘파이 랜덤 고정
torch.manual_seed(SEED) #토치 랜덤 고정
torch.cuda.manual_seed(SEED) #토치 쿠다 시드 고정!
##################################################

DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print('troch:', torch.__version__, '사용 Device:', DEVICE)

#1. data
datasets=np.array([1,2,3,4,5,6,7,8,9,10])

x=np.array([[1,2,3],
           [2,3,4],
           [3,4,5],
           [4,5,6],
           [5,6,7],
           [6,7,8],
           [7,8,9],
           ]) # serial data
y=np.array([4,5,6,7,8,9,10]) # serial data : y값 안 줌

print(x.shape, y.shape) #(7, 3) (7,)

#3차원으로 reshape
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
print(x.shape, y.size()) # torch.Size([7, 3, 1]) torch.Size([7])
#(batch_size, sequence_length, input_size)

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size = 2, shuffle=True)

aaa = iter(train_loader)
bbb = next(aaa)
print(bbb)

#2. 모델
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layer1 = nn.LSTM(
            input_size=1,           # feature 갯수, Tensor에서는 input_dim
            hidden_size=64,         # output_node의 개수, Tensor에서는 unit
            #num_layers=1,          # 디폴트, RNN 은닉층의 레이어의 갯수
            batch_first=True,       # 디폴트 False
            # 원래 이건데(N, 3, 1) False 옵션을 주면 (3, N, 1)
            # 그래서 다시 True 주면 원위치된다. 머리쓰기 귀찮으니깐 그냥 이 옵션 반드시 넣는다.
        )   # (N, 3, 32)
        # self.rnn_layer1 = nn.RNN(1, 64, batch_first=True) #위랑 같음
        # self.fc1 = nn.Linear(3*64, 16)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, h0=None, c0=None): #LSTM 구조를 보면 cell state 가 있음
        # h0 = torch.zeros(1, x.size(0), 32).to(DEVICE) 
        #     # (num_layers, batch_size, hidden_size) hidden_state의 초기값
        # c0 = torch.zeros(1, x.size(0), 32).to(DEVICE) 
            
        x, (hn, cn) = self.lstm_layer1(x, (h0,c0))   # 출력값이 2개(y 값, hidden space)
        # x, _ = self.rnn_layer1(x)
        x = self.relu(x)
        
        # x = x.reshape(-1, 3*64)
        x = x[:, -1, :] #왜 차원이 축소될까? 가장 마지막 timestep꺼만 쓰겠다는 의미
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
model = LSTM().to(DEVICE)

from torchsummary import summary
summary(model, (3, 1))

# from torchinfo import summary
# summary(model, (2,3,1))


#만들어봐
#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []
    
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis.squeeze(-1), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        y_true.append(y_batch.detach().cpu())
        y_pred.append(hypothesis.detach().cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy() #concat
    y_pred = torch.cat(y_pred, dim=0).numpy()
    r2 = r2_score(y_true, y_pred)   
    return total_loss/len(loader), r2

#4. 평가, 예측
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            loss_fin = criterion(hypothesis.squeeze(-1), y_batch)
            total_loss += loss_fin.item()
            
            y_true.append(y_batch.detach().cpu())
            y_pred.append(hypothesis.detach().cpu())
            
    y_true = torch.cat(y_true, dim=0).numpy() #concat
    y_pred = torch.cat(y_pred, dim=0).numpy()            
    r2_fin = r2_score(y_true, y_pred)        
    return total_loss/len(loader), r2_fin

epochs = 500
for epoch in range(1, epochs+1):
    loss, r2 = train(model, criterion, optimizer, train_loader)
    print('epoch:{}, loss:{}, r2:{}'.format(epoch, loss, r2))
    
loss_fin, r2_fin = evaluate(model, criterion, train_loader)
print('최종 loss:', loss_fin)
print('최종 r2:', r2_fin)

x_pred = torch.tensor([[8.0,9.0,10.0]]).reshape(1, 3, 1).to(DEVICE)
result = model(x_pred)
print('[8,9,10]의 예측값:', result.item())

# 최종 loss: 0.00015920841360639315
# 최종 r2: 0.9999574422836304
# [8,9,10]의 예측값: 10.76162338256836

# LSTM
# 최종 loss: 6.879465900055948e-06
# 최종 r2: 0.9999980330467224
# [8,9,10]의 예측값: 10.72603702545166