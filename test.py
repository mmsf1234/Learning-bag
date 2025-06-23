# test.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import *
from model.cnn_model import CatDogCNN


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    # 测试集与 Dataset 相同
    test_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"测试准确率：{100 * correct / total:.2f}%")

if __name__ == '__main__':
    test()
