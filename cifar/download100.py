import os

from torchvision.datasets.cifar import CIFAR100


IMG_DIR = 'data/'


def main():
    """download cifar100 dataset"""
    print("Downloading CIFAR100 dataset.")
    data_path = os.path.join(IMG_DIR, 'CIFAR100')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    cifar = CIFAR100(root=data_path, download=True)
    # cifar.download()
    print("Download complete.")


if __name__ == "__main__":
    main()
