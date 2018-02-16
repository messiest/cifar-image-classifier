import os

from torchvision.datasets.cifar import CIFAR10

IMG_DIR = 'data/'




def download():
    """download cifar100 dataset"""
    print("Downloading CIFAR10 dataset.")
    data_path = os.path.join(IMG_DIR, 'CIFAR10')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    cifar = CIFAR10(root=data_path, download=True)
    # cifar.download()
    print("Download complete.")


def main():
    download()


if __name__ == "__main__":
    main()
