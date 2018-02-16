import os

from torchvision.datasets.cifar import CIFAR100


IMG_DIR = './data'


def main():
    """download cifar100 dataset"""
    print("Downloading CIFAR100 dataset.")
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)
    cifar = CIFAR100(root=IMG_DIR, download=True)
    print("Download complete.")


if __name__ == "__main__":
    main()
