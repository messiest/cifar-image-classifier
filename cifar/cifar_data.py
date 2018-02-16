import torch
import torchvision
import torchvision.transforms as transforms

IMG_DIR = './data'

def main():
    cuda = torch.cuda.is_available()
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    print(cuda)
    train_data = torchvision.datasets.CIFAR10(
        root=IMG_DIR,
        train=True,
        download=True,
        transform=transform_train,
    )

    test_data = torchvision.datasets.CIFAR10(
        root=IMG_DIR,
        train=False,
        download=True,
        transform=transform_test,
    )



if __name__ == "__main__":
    main()