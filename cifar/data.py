import torch
import torchvision
import torchvision.transforms as transforms


IMG_DIR = './data'

def get():
    cuda = torch.cuda.is_available()
    print("Running on GPUs: {}".format(cuda))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.CIFAR10(
        root=IMG_DIR,
        train=True,
        download=True,
        transform=transform_train,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    test_data = torchvision.datasets.CIFAR10(
        root=IMG_DIR,
        train=False,
        download=True,
        transform=transform_test,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    return train_loader, test_loader


def main():
    a, b = get()

    return a, b

if __name__ == "__main__":
    train_loader, test_loader = get()
    print(train_loader, test_loader)
