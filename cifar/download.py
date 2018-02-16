import os

from torchvision.datasets.cifar import CIFAR10, CIFAR100

IMG_DIR = './data'


def download10():
    """download cifar100 dataset"""
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)
        cifar = CIFAR10(root=IMG_DIR, download=True)
        assert(bool(cifar))
        print("download complete")


def download100():
    """download cifar100 dataset"""
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)
    cifar = CIFAR100(root=IMG_DIR, download=True)

    return cifar

class Downloader:
    def __init__(self, dataset=10, n=1, download=False):
        """these values belong to an instance of the class"""
        print("downloading CIFAR-{} data...".format(dataset))
        self.name = "CIFAR{}".format(dataset)

        if download:
            self.download = {
                'CIFAR10': download10(),
                'CIFAR100': download100(),
            }

    def __repr__(self):
        """has to return a string"""
        return self.name


def main():
    # if dataset == '10'
    dl = Downloader(100)


if __name__ == "__main__":
    main()
