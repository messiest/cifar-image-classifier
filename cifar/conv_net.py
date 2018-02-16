import pickle
import os

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

MODEL_PATH = "models/"


class ImageClassifier:

    def __init__(self, img_size=32, device=None):
        self.categories = 10
        self.data = None
        self.labels = None
        # image settings
        self.pixel_depth = 255.0
        self.url = 'https://www.cs.toronto.edu/~kriz/'
        self.last_percent_reported = None
        self.data_root = 'data/cifar-10-batches-py'
        self.num_labels = 10
        # tensorflow graph
        self.saver = None
        self.graph = None
        self.image_size = img_size  # this will have to be flexibly set...
        self.num_channels = 3
        self.num_labels = 10
        self.batch_size = 256
        self.patch_size = 5
        self.depth = 64
        self.num_hidden = 128
        self.logs_path = "/tmp/tf-logs"

        self.train_prediction = None
        self.test_prediction = None

        self.printer = False

        self.writer = None
        self.logits = None
        self.loss = None
        self.optimizer = None

        self.device = None  # Choose device from cmd line. Options: gpu or cpu
        if device == "gpu":
            self.device = "/gpu:0"
        else:
            self.device = "/cpu:0"

    def unpickle(self, file):
        if self.printer:
            print("\t{}".format(file))
        with open(file, 'rb') as file_object:
            return pickle.load(file_object, encoding='latin1')

    def printer_on(self, verbose=False):
        """
            Turn on print messages
        """
        self.printer = True

    def printer_off(self):
        self.printer = False

    def shuffle_data(self, data, labels):
        """
            Helper function used in self.load_data()
        """
        data, _, labels, _, = train_test_split(data, labels, test_size=0.0, random_state=42)
        return data, labels

    def load_data(self, train_batches):
        """
            Used in load_train_data(), and load_test_data()
        """
        data = []
        labels = []
        for data_batch_i in train_batches:
            d = self.unpickle(os.path.join(self.data_root, data_batch_i))
            data.append(d['data'])
            labels.append(np.array(d['labels']))

        # Merge training batches on their first dimension
        data = np.concatenate(data)
        labels = np.concatenate(labels)
        data = data.astype(np.float32)
        labels = (np.arange(self.categories) == labels[:, None]).astype(np.float32)
        length = len(labels)
        data, labels = self.shuffle_data(data, labels)
        data = (data - (self.pixel_depth / 2)) / self.pixel_depth
        data = data.reshape(length, 3, 32, 32)
        return data, labels

    def load_train_data(self, train_batches):
        """
            Load training data
        """
        print('Loading training data...')
        self.train_data, self.train_labels = self.load_data(train_batches)

    def load_test_data(self, test_batches):
        """
            Loading test data
        """
        print('Loading test data...')
        self.test_data, self.test_labels = self.load_data(test_batches)

    def accuracy(self, predictions, labels):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

    def build(self):
        print("Building model...")
        self.graph = tf.Graph()
        self.graph.device(self.device)   # This is what makes it run on GPUs

        with self.graph.as_default():
            self.writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())

            # input data
            self.tf_train_data = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size, self.image_size,self.num_channels), name="training_data")
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels), name="training_labels")
            self.tf_test_data = tf.constant(self.test_data.transpose(0, 2, 3, 1))

            # variables

            # instantiate weight and bias tensors

            # layer 1 - convolution  # TODO (@messiest) adjust this so that the image size is 128
            weights_1 = tf.Variable(
                tf.truncated_normal([self.patch_size, self.patch_size, self.num_channels, self.depth], stddev=0.1))
            biases_1 = tf.Variable(tf.zeros([self.depth]))
            print("\tLayer 1: w={}, b={}".format(weights_1.shape, biases_1.shape))

            # layer 2 - convolution
            weights_2 = tf.Variable(
                tf.truncated_normal([self.patch_size, self.patch_size, self.depth, self.depth], stddev=0.1))
            biases_2 = tf.Variable(tf.constant(1.0, shape=[self.depth]))
            print("\tLayer 2: w={}, b={}".format(weights_2.shape, biases_2.shape))

            # layer 3 - convolution  # TODO (@messiest) You need to get this layer working!!!
            # weights_3 = tf.Variable(
            #     tf.truncated_normal([self.patch_size, self.patch_size, self.depth, self.depth],stddev=0.1))
            # biases_3 = tf.Variable(tf.constant(1.0, shape=[self.depth]))
            # print("\tLayer 3: w={}, b={}".format(weights_3.shape, biases_3.shape))

            # layer 4 - hidden
            weights_4 = tf.Variable(
                tf.truncated_normal(
                    [self.image_size // 4 * self.image_size // 4 * self.depth, self.num_hidden], stddev=0.1))
            biases_4 = tf.Variable(tf.constant(1.0, shape=[self.num_hidden]))
            print("\tLayer 4: w={}, b={}".format(weights_4.shape, biases_4.shape))

            # layer 5 - hidden
            weights_5 = tf.Variable(
                tf.truncated_normal(
                    [self.num_hidden, self.num_hidden], stddev=0.1))
            biases_5 = tf.Variable(tf.constant(1.0, shape=[self.num_hidden]))
            print("\tLayer 5: w={}, b={}".format(weights_5.shape, biases_5.shape))

            # layer 6 - output
            layer6_weights = tf.Variable(tf.truncated_normal([self.num_hidden, self.num_labels], stddev=0.1))
            layer6_biases = tf.Variable(tf.constant(1.0, shape=[self.num_labels]))

            # model
            def model(data):
                """
                    Convolutional Neural Network
                    - layer 1: convolution
                    - layer 2: convolution
                    - layer 4: hidden
                    - layer 5: hidden
                    - layer 6: output

                :param data:
                :type data:
                :return:
                :rtype:
                """
                # layer 1 - convolution
                conv = tf.nn.conv2d(data, weights_1, [1, 1, 1, 1], padding='SAME', name='1-conv')
                conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='1-maxpool')
                hidden = tf.nn.relu(conv + biases_1, name='1-relu')
                hidden = tf.nn.dropout(hidden, 0.75, name='1-dropout')

                # layer 2 - convolution
                conv = tf.nn.conv2d(hidden, weights_2, [1, 1, 1, 1], padding='SAME', name='2-conv')
                conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='2-maxpool')
                hidden = tf.nn.relu(conv + biases_2, name='2-relu')
                hidden = tf.nn.dropout(hidden, 0.75, name='2-dropout')

                # layer 4 - hidden
                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(reshape, weights_4) + biases_4)
                hidden = tf.nn.dropout(hidden, 0.75, name='4-dropout')

                # layer 5 - hidden
                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1]])
                hidden = tf.nn.relu(tf.matmul(reshape, weights_5) + biases_5)
                hidden = tf.nn.dropout(hidden, 0.75, name='5-dropout')

                # layer 6 - output
                output = tf.matmul(hidden, layer6_weights) + layer6_biases

                return output

            # training computation
            self.logits = model(self.tf_train_data)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_train_labels, logits=self.logits))

            # optimizer
            self.optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(self.loss)

            # predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(self.logits)
            self.test_prediction = tf.nn.softmax(model(self.tf_test_data))

        print("Model built.")

    def train(self, num_steps=100):
        print("Training model...")
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()     # initialize the variables

            try:  # load saved model
                self.saver = tf.train.import_meta_graph(MODEL_PATH + '.meta')
                self.saver.restore(session, MODEL_PATH)
                print("Existing Model Loaded")

            except:
                self.saver = tf.train.Saver(max_to_keep=1)  # used to save the model

                for step in range(num_steps):
                    offset = (step * self.batch_size) % (self.train_labels.shape[0] - self.batch_size)
                    batch_data = self.train_data[offset:(offset + self.batch_size), :, :, :]
                    batch_data = batch_data.transpose(0, 2, 3, 1)
                    batch_labels = self.train_labels[offset:(offset + self.batch_size), :]
                    feed_dict = {self.tf_train_data: batch_data, self.tf_train_labels: batch_labels}
                    _, l, predictions = session.run(
                        [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                    if (step % 100 == 0) and self.printer:
                        print('Step: {}'.format(step))
                        print('- loss: {:.2f}'.format(l))
                        print('- accuracy: {:.2f}%\n'.format(self.accuracy(predictions, batch_labels)))

                self.saver.save(session, "models/")

            tp = self.test_prediction.eval()

            images = self.test_data.transpose(0, 2, 3, 1)

            print("Model trained.")
            if self.printer:
                print('Test accuracy: {:.2f}%'.format(self.accuracy(tp, self.test_labels)))


if __name__ == '__main__':
    cnn = ImageClassifier(img_size=64)
    cnn.printer_on()

    # data loading
    cnn.load_train_data(["data_batch_{}".format(i) for i in range(1, 6)])
    cnn.load_test_data(["test_batch"])

    # build tensorflow graph
    cnn.build()
    cnn.train()
