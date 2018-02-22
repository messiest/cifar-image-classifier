import os
import numpy as np
import cv2

import time


def get_label_vectors():
    """
     wgenerate label vectors from collected images

    :return: dictionary of labels and label vectors
    :rtype: dict
    """
    print("Retrieving label vectors...")
    label_dict = {}  # instantiate dict for labels:vectors
    # read in image files
    categories = sorted((c for c in os.listdir('images/') if c[0] != '.'))
    x = np.zeros(len(categories))                                           # zero vector of number of categories
    for i, c in enumerate(categories):                                      # get index and category for images
        y = x.copy()                                                        # use copy of x
        y[i] = 1                                                            # set label index to true
        label_dict[c] = y                                                   # create label:vector
        del y

    return label_dict


def load_image_data():
    """
    load image data from collected categories

    :return: image data vectors
    :rtype: numpy.ndarray
    """
    label_dict = get_label_vectors()
    print("Retrieved label vectors.")
    paths = (c for c in label_dict.keys())
    files = []
    labels = []
    for p in paths:
        dir = 'images/{}/'.format(p)
        print(dir)
        for f in os.listdir(dir):
            files.append(dir + f)
            labels.append(label_dict[p])
    print("Done")
    images = (cv2.imread(f).flatten() for f in files)
    data = np.array([i for i in images])

    return data


def load_image_labels():
    """
    load image data from collected categories

    :return: label vectors
    :rtype: numpy.ndarray
    """
    print("Loading image labels...")
    label_dict = get_label_vectors()
    print("Retrieved vector names.")
    categories = (c for c in os.listdir('images/') if c[0] != '.')  # ignore
    labels = []  # instantiate list for image labels
    for i in categories:
        path = 'images/{}/'.format(i)  # define path to category folder
        for _ in os.listdir(path):  # get images from category folder
            labels.append(label_dict[i])  # append label vector
    labels = np.array(labels)  # convert lists to array
    print("Done.")

    return labels

def main():
    """
        main method of image_mgmt.py

    """
    # labels = load_image_labels()
    data = load_image_data()
    # print(labels.shape)
    print(data.shape)


if __name__ == "__main__":
    main()
