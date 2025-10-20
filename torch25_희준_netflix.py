<<<<<<< HEAD
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import accuracy_score, r2_score

random.seed(333)
np.random.seed(333)
torch.manual_seed(333)
torch.cuda.manual_seed(333)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

#1. 데이터
path = 'c:/Study26/_data/kaggle/netflix/' 
train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
print(train_csv) #[967 rows x 6 columns] 
# [967 rows x 3 columns] -> (n, 30, 3)으로 바꿔야 rnn에 돌릴 수 있어
# Volume: 거래량, Close: 종가
print(train_csv.info()) # 결측치 없음
print(train_csv.describe())

# import matplotlib.pyplot as plt
# data = train_csv.iloc[:, 1:4] # 행, 열 순서
# data['종가'] = train_csv['Close']
# print(data)

# hist = data.hist()
# plt.show()

from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

class Custom_Dataset(Dataset):
    def __init__(self, df, timesteps=30):
        self.train_csv = df
        
        self.x = self.train_csv.iloc[:, 1:4].values
        self.x = (self.x - np.min(self.x, axis=0))/ \
            (np.max(self.x, axis=0) - np.min(self.x, axis=0)) # MinMaxScaler
        
        self.y = self.train_csv['Close'].values
        self.timesteps = timesteps
        
    
    # (10,1)->(8,3,1) 즉, 전체 행 - timesteps   
    # (967, 3) -> (n, 30, 3)
    def __len__(self):
        return len(self.x) - self.timesteps       #행 - Timesteps

    def __getitem__(self, idx):
        x = self.x[idx : idx+self.timesteps]    # x[idx: idx+timesteps]              
        y = self.y[idx+self.timesteps]          # y[idx+timesteps]
        return x, y

custom_dataset = Custom_Dataset(df=train_csv, timesteps=30)

train_loader = DataLoader(custom_dataset, batch_size=32)

for batch_idx, (xb, yb) in enumerate(train_loader):
    print("================배치: ", batch_idx, '========================')
    print('x: ', xb.shape)
    print('y: ', yb.shape)
    break

# ================배치:  0 ========================
# x:  torch.Size([32, 30, 3])
# y:  torch.Size([32])

#2. 모델
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(input_size=3,
                          hidden_size=64,
                          num_layers=3,     #데이터가 좀 되면 3-5로 해줘
                          batch_first=True,
                          ) # (n, 30, 3) 로 들어와서 -> (n, 30, 64)
        # self.fc1 = nn.Linear(in_features=30*64, out_features=32)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(32,1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.rnn(x)
        
        # x = x.reshape(-1, 30 * 64)
        x = x[:, -1, :]
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = RNN().to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()

from torch.optim import Adam
optim = Adam(params=model.parameters(), lr=0.001)

# import tqdm
from tqdm import tqdm

for epoch in range(1, 201):
    iterator = tqdm(train_loader)
    for x, y in iterator:
        optim.zero_grad()
        
        hypothesis = model(x.to(DEVICE))
        loss = nn.MSELoss()(hypothesis, y)
        
        loss.backward()
        optim.step()
        
        # iterator.set_description()
        
        

exit()
## save ##
save_path = './_save/torch/'
torch.save(model.state_dict(), save_path + 't25_netflix.pth')

        
# def train(model, criterion, optimizer, loader):
#     model.train()
#     total_loss = 0
#     y_true = []
#     y_pred = []
    
#     for x_batch, y_batch in loader:
#         x_batch = x_batch.to(DEVICE)
#         y_batch = y_batch.to(DEVICE)
        
#         optimizer.zero_grad()
#         hypothesis = model(x_batch)
#         loss = criterion(hypothesis.squeeze(-1), y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
        
#         y_true.append(y_batch.detach().cpu())
#         y_pred.append(hypothesis.detach().cpu())
    
#     y_true = torch.cat(y_true, dim=0).numpy() #concat
#     y_pred = torch.cat(y_pred, dim=0).numpy()
#     r2 = r2_score(y_true, y_pred)   
#     return total_loss/len(loader), r2

# #4. 평가, 예측
# def evaluate(model, criterion, loader):
#     model.eval()
#     total_loss = 0
#     y_true = []
#     y_pred = []
    
#     with torch.no_grad():
#         for x_batch, y_batch in loader:
#             x_batch = x_batch.to(DEVICE)
#             y_batch = y_batch.to(DEVICE)
            
#             hypothesis = model(x_batch)
#             loss_fin = criterion(hypothesis.squeeze(-1), y_batch)
#             total_loss += loss_fin.item()
            
#             y_true.append(y_batch.detach().cpu())
#             y_pred.append(hypothesis.detach().cpu())
            
#     y_true = torch.cat(y_true, dim=0).numpy() #concat
#     y_pred = torch.cat(y_pred, dim=0).numpy()            
#     r2_fin = r2_score(y_true, y_pred)        
#     return total_loss/len(loader), r2_fin

# epochs = 100
# for epoch in range(1, epochs+1):
#     loss, r2 = train(model, criterion, optimizer, train_loader)
#     print('epoch:{}, loss:{}, r2:{}'.format(epoch, loss, r2))
    
# loss_fin, r2_fin = evaluate(model, criterion, train_loader)
# print('최종 loss:', loss_fin)
# print('최종 r2:', r2_fin)
# exit()

# x_pred = torch.tensor([[8.0,9.0,10.0]]).reshape(1, 3, 1).to(DEVICE)
# result = model(x_pred)
=======
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import accuracy_score, r2_score

random.seed(333)
np.random.seed(333)
torch.manual_seed(333)
torch.cuda.manual_seed(333)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

#1. 데이터
path = 'c:/Study26/_data/kaggle/netflix/' 
train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
print(train_csv) #[967 rows x 6 columns] 
# [967 rows x 3 columns] -> (n, 30, 3)으로 바꿔야 rnn에 돌릴 수 있어
# Volume: 거래량, Close: 종가
print(train_csv.info()) # 결측치 없음
print(train_csv.describe())

# import matplotlib.pyplot as plt
# data = train_csv.iloc[:, 1:4] # 행, 열 순서
# data['종가'] = train_csv['Close']
# print(data)

# hist = data.hist()
# plt.show()

from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

class Custom_Dataset(Dataset):
    def __init__(self, df, timesteps=30):
        self.train_csv = df
        
        self.x = self.train_csv.iloc[:, 1:4].values
        self.x = (self.x - np.min(self.x, axis=0))/ \
            (np.max(self.x, axis=0) - np.min(self.x, axis=0)) # MinMaxScaler
        
        self.y = self.train_csv['Close'].values
        self.timesteps = timesteps
        
    
    # (10,1)->(8,3,1) 즉, 전체 행 - timesteps   
    # (967, 3) -> (n, 30, 3)
    def __len__(self):
        return len(self.x) - self.timesteps       #행 - Timesteps

    def __getitem__(self, idx):
        x = self.x[idx : idx+self.timesteps]    # x[idx: idx+timesteps]              
        y = self.y[idx+self.timesteps]          # y[idx+timesteps]
        return x, y

custom_dataset = Custom_Dataset(df=train_csv, timesteps=30)

train_loader = DataLoader(custom_dataset, batch_size=32)

for batch_idx, (xb, yb) in enumerate(train_loader):
    print("================배치: ", batch_idx, '========================')
    print('x: ', xb.shape)
    print('y: ', yb.shape)
    break

# ================배치:  0 ========================
# x:  torch.Size([32, 30, 3])
# y:  torch.Size([32])

#2. 모델
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(input_size=3,
                          hidden_size=64,
                          num_layers=3,     #데이터가 좀 되면 3-5로 해줘
                          batch_first=True,
                          ) # (n, 30, 3) 로 들어와서 -> (n, 30, 64)
        # self.fc1 = nn.Linear(in_features=30*64, out_features=32)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(32,1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.rnn(x)
        
        # x = x.reshape(-1, 30 * 64)
        x = x[:, -1, :]
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = RNN().to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()

from torch.optim import Adam
optim = Adam(params=model.parameters(), lr=0.001)

# import tqdm
from tqdm import tqdm

for epoch in range(1, 201):
    iterator = tqdm(train_loader)
    for x, y in iterator:
        optim.zero_grad()
        
        hypothesis = model(x.to(DEVICE))
        loss = nn.MSELoss()(hypothesis, y)
        
        loss.backward()
        optim.step()
        
        # iterator.set_description()
        
        

exit()
## save ##
save_path = './_save/torch/'
torch.save(model.state_dict(), save_path + 't25_netflix.pth')

        
# def train(model, criterion, optimizer, loader):
#     model.train()
#     total_loss = 0
#     y_true = []
#     y_pred = []
    
#     for x_batch, y_batch in loader:
#         x_batch = x_batch.to(DEVICE)
#         y_batch = y_batch.to(DEVICE)
        
#         optimizer.zero_grad()
#         hypothesis = model(x_batch)
#         loss = criterion(hypothesis.squeeze(-1), y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
        
#         y_true.append(y_batch.detach().cpu())
#         y_pred.append(hypothesis.detach().cpu())
    
#     y_true = torch.cat(y_true, dim=0).numpy() #concat
#     y_pred = torch.cat(y_pred, dim=0).numpy()
#     r2 = r2_score(y_true, y_pred)   
#     return total_loss/len(loader), r2

# #4. 평가, 예측
# def evaluate(model, criterion, loader):
#     model.eval()
#     total_loss = 0
#     y_true = []
#     y_pred = []
    
#     with torch.no_grad():
#         for x_batch, y_batch in loader:
#             x_batch = x_batch.to(DEVICE)
#             y_batch = y_batch.to(DEVICE)
            
#             hypothesis = model(x_batch)
#             loss_fin = criterion(hypothesis.squeeze(-1), y_batch)
#             total_loss += loss_fin.item()
            
#             y_true.append(y_batch.detach().cpu())
#             y_pred.append(hypothesis.detach().cpu())
            
#     y_true = torch.cat(y_true, dim=0).numpy() #concat
#     y_pred = torch.cat(y_pred, dim=0).numpy()            
#     r2_fin = r2_score(y_true, y_pred)        
#     return total_loss/len(loader), r2_fin

# epochs = 100
# for epoch in range(1, epochs+1):
#     loss, r2 = train(model, criterion, optimizer, train_loader)
#     print('epoch:{}, loss:{}, r2:{}'.format(epoch, loss, r2))
    
# loss_fin, r2_fin = evaluate(model, criterion, train_loader)
# print('최종 loss:', loss_fin)
# print('최종 r2:', r2_fin)
# exit()

# x_pred = torch.tensor([[8.0,9.0,10.0]]).reshape(1, 3, 1).to(DEVICE)
# result = model(x_pred)
>>>>>>> 16a711e (initialize torch repo)
# print('[8,9,10]의 예측값:', result.item())