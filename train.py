import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import *
from data.dataset_loader import get_dataloaders
from model.cnn_model import CatDogCNN

loss_list = []
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogCNN().to(device)
    dataloader = get_dataloaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir="runs/cat_dog_experiment")  # 创建 TensorBoard 日志目录

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

        writer.add_scalar("Loss/train", avg_loss, epoch)

    writer.close()
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == '__main__':
    train()
