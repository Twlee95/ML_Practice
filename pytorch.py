import torch
x = torch.tensor(data=[2.0, 3.0], requires_grad=True)
y = x**2
z = 2*x+3

target = torch.tensor([3.0, 4.0])
loss = torch.sum(torch.abs(z-target))
##loss = torch.sqrt(torch.sum((z-target)**2))
loss.backward()

print(x.grad, y.grad, z.grad)

## module import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

## dataset
num_data = 100
num_epoch = 500
x = init.uniform_(torch.Tensor(num_data, 1), -10, 10)
noise = init.normal_(torch.FloatTensor(num_data, 1), std=1)
y = 2*x+3
y_noise = 2*(x+noise)+3

## model
model = nn.Linear(1, 1)
loss_func = nn.L1Loss()

##optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

label = y_noise
for i in range(num_epoch):
    optimizer.zero_grad()
    output = model(x)

    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(loss.data)

param_list = list(model.parameters())
print(param_list[0].item(), param_list[1].item())


import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.array(x), np.array(y_noise), 'or')
plt.plot(np.array(x),model())

np.array(range(num_epoch)),loss.data