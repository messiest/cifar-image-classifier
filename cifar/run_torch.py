import time
import sys

import torch
from torch.autograd import Variable

from googlenet import GoogLeNet


try:
    device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
    if device_name == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"
except:
    device_name = "/cpu:0"


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
