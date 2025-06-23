import torch
from PIL import Image
from torchvision import transforms

from config import *
from model.cnn_model import CatDogCNN

classes = ['Cat', 'Dog']

def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        print(f"预测结果：{classes[predicted.item()]}")

if __name__ == '__main__':
    img_path = 'E:/Python/猫狗识别/test/459.jpg'  # 待预测图片路径
    predict_image(img_path)
