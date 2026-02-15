import torch
import torchvision
import torchvision.transforms as transforms
def get_mnist_loader(batch_size=1,shuffle=True):
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
    dataset=torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    loader=torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle =shuffle
    )
    return loader