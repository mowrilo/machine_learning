import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

x = torch.Tensor([[0,0],[0,1],[1,0],[1,1]])
y = torch.Tensor([0,1,1,0])
y = y.reshape(4,1)

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.lin1 = nn.Linear(2,2)
        self.act1 = nn.Tanh()
        self.lin2 = nn.Linear(2,1)
        self.act2 = nn.Sigmoid()
    def forward(self, x):
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        return x

net = MLP()
y_hat = net.forward(x)
print("Predictions before training:",(y_hat > .5).t().numpy())

learning_rate = .05
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()

n_epochs = 2000
net.train()
for i in range(n_epochs):
    y_hat = net.forward(x)
    loss = loss_func(y_hat, y)
    loss.backward(loss)
    optimizer.step()
net.eval()

y_hat = net.forward(x)
print("Predictions before training:",(y_hat > .5).t().numpy())

