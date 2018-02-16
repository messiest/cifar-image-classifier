import time
import sys

import torch
from torch.autograd import Variable

from models.googlenet import GoogLeNet


cuda = torch.cuda.is_available()  # Hopefully you can you use GPUs?


def main(timer=True):
    print('Running...')
    start = time.time()  # start timer
    net = GoogLeNet()
    net.zero_grad()
    print(net)
    params = list(net.parameters())
    print(len(params))
    _input = Variable(torch.randn(1, 3, 32, 32))  # create input vector
    out = net(_input)  # run the input vector through the network
    print('out\n', out)

    if timer:
        print('Complete.', 'Run time: {:.2f} sec.'.format(time.time() - start))
    else:
        print('Complete.')


if __name__ == "__main__":
    main(timer=True)
