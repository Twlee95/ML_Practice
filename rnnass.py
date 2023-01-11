import numpy as np
import matplotlib.pyplot as plt
## generating dataset
num_data = 2400
t = np.linspace(0.0, 100.0, num_data)
y = np.sin(t)+np.sin(2*t)
e = np.random.normal(0, 0.1, num_data)
y = y+e
plt.plot(t, y+e)

seq_len = 10
X = []
y_true = []
for i in range(len(t)-seq_len):
    X.append(y[i:i+seq_len])
    y_true.append(y[i+seq_len])
X = np.array(X)
y_true = np.array(y_true)

X = np.swapaxes(X, 0, 1)
X = np.expand_dims(X, axis=2)
for x in X:
    print(x)
    break
X[0].shape

X[9].shape



import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, batch_size):
        super(RNN, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.batch_size = batch_size

        self.u = nn.Linear(self.in_dim, self.hid_dim, bias=False)
        self.w = nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        self.v = nn.Linear(self.hid_dim, self.out_dim, bias=False)
        self.act = nn.Tanh()

        self.hidden = self.init_hidden() ## hidden vector가 0 벡터로 초기화된 상태로 시작

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.zeros(batch_size, self.hid_dim)

    def forward(self, x):
        print(self.hid_dim[0])
        h = self.u(x)+self.w(self.hidden)
        h = self.act(h)
        y = self.v(h)
        return y, h

import torch.optim as optim
model = RNN(1, 1, 10, 2390)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
epoch = 10

## train
for i in range(epoch): #epoch수 만큼 학습을 반복
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    for x in X:  ## 이 반복문 까지가 한 epoch
        x = torch.Tensor(x).float()
        y_true = torch.Tensor(y_true).float()
        y_pred, hidden = model(x)
    loss = loss_fn(y_pred.view(-1), y_true.view(-1)) ## .view(-1)을 하는것은 loss 계산시 dim을 맞추기 위해서
    loss.backward() ## .backward()를 사용하여 loss값을 모형에 학습시킴
    optimizer.step()
    print(loss.item())

print(X.shape)
test_X = np.expand_dims(X[:, 0, :], 1)
print(test_X.shape)


list_y_pred = []

model.eval()
with torch.no_grad():
    model.hidden = model.init_hidden(batch_size=1)

    for x in X:  ## 이 반복문 까지가 한 epoch
       x = torch.Tensor(x).float()
       y_true = torch.Tensor(y_true).float()
       model.hidden = hidden

    list_y_pred.append(y_pred.view(-1).item())

    temp_X = list()
    temp_X = tset_X
    for i in range(2389):
        model.hidden = model.init_hidden(batch_size=1)

        for x in temp_X:
            y_pred, hidden = model(x)
            model.hidden = hidden
        y_pred, hidden = model(y_pred)
        list_y_pred.append(y_pred.view(-1).item())
    print(list_y_pred)