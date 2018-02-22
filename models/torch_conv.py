import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F  # torch functions
import torch.optim as optim  # torch optimization


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))  # 1 image, 6 output channels, 5x5 convolution
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        # hidden layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # layer 1 activation
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        forward must be overwritten in torch model class
        """
        # conv layers
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # add pooling layer
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))  # view manipulates shape

        # fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


if __name__ == "__main__":
    net = LeNet()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print('conv1 weight\n', params[0].size())  # conv1's .weight

    _input = Variable(torch.randn(1, 1, 32, 32))  # create input vector
    out = net(_input)  # run the input vector through the network
    print('out\n', out)

    net.zero_grad()
    out.backward(torch.randn(1, 10))

    output = net(_input)
    target = Variable(torch.arange(1, 11))  # a dummy target, for example
    criterion = nn.MSELoss()

    loss = criterion(output, target)  # loss function docs: http://pytorch.org/docs/master/nn.html
    print('loss\n', loss)

    print(loss.grad_fn)

    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    net.zero_grad()  # zeroes the gradient buffers of all parameters
    print('conv1.bias.grad before backward propagation')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    # gradient descent
    learning_rate = 0.01
    for f in net.parameters():  # this
        f.data.sub_(f.grad.data * learning_rate)

    params = list(net.parameters())
    print(len(params))
    print('conv1 weight\n', params[0].size())  # conv1's .weight

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    for i in range(10):
        optimizer.zero_grad()  # zero the gradient buffers
        output = net(_input)
        loss = criterion(output, target)  # mse loss
        loss.backward()  # back propagate
        optimizer.step()  # do the update
