import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.optim as optim


## transforms.Compose 파이토치에있는 이미지 변환 모듈
## 이미지 파일을 텐서로 쉽게 바꿔주는 역할을 함
## rgb 값을 "transforms.ToTensor()" 에 넣으면 0~1로 변환해줌 (어두우면 0, 밝으면 1    )
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# transforms.Normalize((R, G, B), (0.5, 0.5, 0.5))
##                        평균          표준편차
## >>> 결론 : Normalize를 각 채널별로 시켜줬다.
print(os.getcwd())

### iter 실행시 오류가 뜬다면 num_worker 를 삭제하면 된다
## Delete the num_workers of the data loader, namely
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
## 총 60000장 40000 10000 10000
trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                        shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')





# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


## type : 모델이 tensor로 잘 불러와졌는지 확인하기위함.
## images.shape : <class 'torch.Tensor'> torch.Size([4, 3, 32, 32])
#                                     [batch size(각 하나의 사진), channel, 가로,세로]
print(type(images), images.shape)

## iter 한 데이터set에 .next()를  하게되면(dataiter.next()) 자동으로 다음 batch 를 보여주게된다/.
# # print(labels) >>>> tensor([2, 9, 1, 8]) : classes 에서 2번째 bird, 9번쨰 'truck' 등등을 나타낸다.
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(type(labels), labels.shape, labels)


## MODEL
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, n_layer, act):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.act = act

        self.fc = nn.Linear(self.in_dim, self.hid_dim)
        ## 그냥 list가 아니고 모듈 리스트를 쓰는 이유는
        ## 이렇게 하면 optimizer가 parameter에 잘 접근할수 있게 해준다
        self.linears = nn.ModuleList()

        ## num_layer를 hidden 만으로 생각한다면 그대로 쓰면되지만
        ## input layer 까지 포함한 수로 생각한다면 -1 을 해주면 된다
        for i in range(self.n_layer - 1):
            self.linears.append(nn.Linear(self.hid_dim, self.hid_dim))
        self.fc2 = nn.Linear(self.hid_dim, self.out_dim)

        if self.act == 'relu':
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc(x))
        for fc in self.linears:
            x = self.act(fc(x))
        x = self.fc2(x)   ## 마지막에는 relu를 넣지 않는게 좋다, (값이 0이되면 classification에 문제가 생길 수 있다.)
        return x




## train
net = MLP(3072, 10, 100, 4, 'relu')
print(net)





## test
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))





##EXPERIMENT

def experiment(args):
    net = MLP(args.in_dim,args.out_dim,args.hid_dim, args.n_layer, args.act)
    ## net.to('cuda:0') or net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mm)

    for epoch in range(args.epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        train_loss =0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.view(-1, 3072)
            ## inputs = inputs.cuda()
            ## labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        ## val
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images = images.view(-1, 3072)
                ## images = images.cuda()
                ## labels = labels.cuda()
                outputs = net(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_loss = val_loss / len(valloader)
            val_acc = 100 * correct / total
            print('epoch :{},train_loss:{},val_loss:{},val_acc:{}'.format(epoch, train_loss, val_loss, val_acc))
    ## test acc
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(-1, 3072)
            ## images = images.cuda()
            ## labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = correct/total*100
    return train_loss, val_loss, val_acc, test_acc


import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args('')

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

args.n_layer = 5
args.in_dim = 3072
args.out_dim = 10
args.hid_dim = 100
args.lr = 0.001
args.mm = 0.9
args.act = 'relu'
args.epoch =2


experiment(args)


list_var1 = [4, 5, 6]
list_var2 = [50, 100, 150]

for var1 in list_var1:
    for var2 in list_var2:
        args.n_layer = var1
        args.hid_dim = var2
        result = experiment(args)
        print(result)