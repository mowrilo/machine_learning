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
        self.lin1 = nn.Linear(2,2,bias=True)
        #self.act1 = nn.Tanh()
        self.lin2 = nn.Linear(2,1,bias=True)
        #self.act2 = nn.Sigmoid()
    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        return x

torch.manual_seed(2)
net = MLP()
#nn.init.constant_(net.lin1.weight,0.)

for c in net.children():
    print(c)
    nn.init.normal_(c.weight,0.,.1)
    nn.init.normal_(c.bias,0.,.1)
#nn.init.constant_(net.lin1.bias,0.)
#nn.init.constant_(net.lin2.weight,0.)
#nn.init.constant_(net.lin2.bias,0.)

y_hat = net.forward(x)
print("Predictions before training:",(y_hat > .5).t().numpy())

learning_rate = .1
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()

n_epochs = 10000
net.train()
for i in range(n_epochs):
    optimizer.zero_grad()
    y_hat = net.forward(x)
    loss = torch.pow(y_hat - y,2).sum()#loss_func(y_hat, y)
    #print(loss)
    loss.backward(loss)
    optimizer.step()
#    print(net.lin2.weight.data)
net.eval()

y_hat = net.forward(x)
print("Predictions after training:",(y_hat > .5).t().numpy())

