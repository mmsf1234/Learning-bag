from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(data_dir, batch_size=32, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
