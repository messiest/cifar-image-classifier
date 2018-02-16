import time
import sys

from googlenet import GoogLeNet
from cifar.conv_net import ImageClassifier

try:
    device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
    if device_name == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"
except:
    device_name = "/cpu:0"


def main(printer=True, timer=True):
    print('Running...')
    start = time.time()
    cnn = ImageClassifier(device=device_name)  # instantiate classifier
    if printer:
        cnn.printer_on()  # turn on printing
    cnn.load_train_data(["data_batch_{}".format(i) for i in range(1, 6)])  # load training data
    cnn.load_test_data(["test_batch"])  # load test data
    cnn.build()  # build TensorFlow model
    cnn.train(num_steps=5000)  # train TensorFlow model
    if timer:
        print('Complete.', 'Run time: {:.2f} sec.'.format(time.time() - start)) #
    else:
        print('Complete.')


if __name__ == "__main__":
    try:
        main(printer=sys.argv[1], timer=sys.argv[2])

    except IndexError:
        main(printer=True, timer=True)
