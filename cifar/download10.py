import os

from torchvision.datasets.cifar import CIFAR10

IMG_DIR = './data'




def download():
    """download cifar100 dataset"""
    print("Downloading CIFAR10 dataset.")
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)
    cifar = CIFAR10(root=IMG_DIR, download=True)
    print("Download complete.")


def main():
    download()


if __name__ == "__main__":
    main()
