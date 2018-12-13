import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch import optim

print("Loading and transforming images...",end='')
data_folder = './data'
images_train = datasets.MNIST(root=data_folder, train=True, download=True, transform=None)
images_test = datasets.MNIST(root=data_folder, train=False, download=True, transform=None)

n_train = len(images_train)
n_test = len(images_test)
image_size = 28*28
n_digits = 10

data_train = torch.zeros([n_train,image_size]).float()
lab_train = torch.zeros([n_train, n_digits]).float()
data_test = torch.zeros([n_test,image_size]).float()
lab_test = torch.zeros([n_test, n_digits]).float()

for idx in range(n_train):
    im = images_train[idx][0]
    data_train[idx] = torch.from_numpy(np.array(im).reshape(1,image_size))
    lab_train[idx,images_train[idx][1].item()] = 1.

for idx in range(n_test):
    im_t = images_test[idx][0]
    data_test[idx] = torch.from_numpy(np.array(im_t).reshape(1,image_size))
    lab_test[idx,images_test[idx][1].item()] = 1.

data_train /= 255
data_test /= 255

print(" Done!\n")

class NNet(nn.Module):
    def __init__(self,n_inputs,n_outputs):
        super(NNet,self).__init__()
        self.lay1 = nn.Linear(n_inputs,500,bias=True)
        self.lay2 = nn.Linear(500,50,bias=True)
        self.lay3 = nn.Linear(50,n_outputs,bias=True)
        self.final = nn.Softmax(dim=1)
    def forward(self,x):
        x = F.relu(self.lay1(x))
        x = F.relu(self.lay2(x))
        x = self.lay3(x)
        x = self.final(x)
        return x

torch.manual_seed(5)
net = NNet(image_size,n_digits)
#optimizer = optim.SGD(net.parameters(), lr=.05)#, momentum=.5, weight_decay=.8)
optimizer = optim.RMSprop(net.parameters(),lr=.001)
#optimizer = optim.Adam(net.parameters())
loss_func = nn.BCELoss()

n_epochs = 101
net.train()

print("Training...")
for ix in range(n_epochs):
    optimizer.zero_grad()
    y_hat = net.forward(data_train)
    loss = loss_func(y_hat,lab_train)
    loss.backward(loss)
    optimizer.step()
    if ix % 10 == 0:
        print("\tEpoch:", ix, "\n\t\t Loss:", loss.item())
print("Done training.")
net.eval()

n_right = 0
print("\nTesting...")

y = net.forward(data_test)
final_loss = loss_func(y,lab_test)

for ix in range(n_test):
    y_hat = y[ix]
    lab = lab_test[ix]
    n_right += 1*(y_hat.argmax().item() == lab.argmax().item())

print("\tFinal loss: %.5f\n\tAccuracy: %.3f%%" % (final_loss.item(), 100*n_right/n_test))
