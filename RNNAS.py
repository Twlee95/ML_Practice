import numpy as np
import matplotlib.pyplot as plt

num_data = 2400
t = np.linspace(0.0, 100.0, num_data)
y = np.sin(t)+np.sin(2*t)
e = np.random.normal(0, 0.1, num_data)

plt.plot(t, y)

seq_len = 10
X = []
y_true = []
for i in range(len(t)-seq_len):
    X.append(y[i:i+seq_len])
    y_true.append(y[i+seq_len])
X = np.array(X)
y_true = np.array(y_true)
type(X)
X[0].shape
## RNN 차원을 맞출 시
## [seq len, batch size, input_dim]
## for 문을 돌리기 쉬워서 seq len이 가장 앞에 있다
##[2390,1]>>>>[10, 2390, 1]처럼 되기를 원함

X = np.swapaxes(X, 0, 1)
X = np.expand_dims(X, axis=2)
print(X.shape)
print(y_true.shape)


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

    def forward(self, xx):
        h = self.act(self.u(xx) + self.w(self.hidden))
        y_p = self.v(h)
        return y_p, h

import torch.optim as optim
model = RNN(1, 1, 10, 2390)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
epoch = 10


for i in range(epoch):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    #X[:, 0, :]
    #X[:, 1, :]
    #X[:, 5, :]
    #y_true
    for i in range(2390):
        x = X[:, i, :]
        y_true = np.array(y_true[i,])
        x = torch.Tensor(x).float()
        y_true = torch.Tensor(y_true).float()
        y_pre, hidden = model(x)
        y_pre.shape
        model.hidden = hidden

    loss = loss_fn(y_pre.view(-1), y_true.view(-1))
    loss.backward(retain_graph=True)
    optimizer.step()
    print(loss.item())
