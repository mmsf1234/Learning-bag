import tkinter as tk
from tkinter import filedialog

import torch
from PIL import Image, ImageTk
from torchvision import transforms

from config import *
from model.cnn_model import CatDogCNN

# 设置类别名称（与训练数据集目录一致）
classes = ['Cat', 'Dog']

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatDogCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# 创建主窗口
window = tk.Tk()
window.title("猫狗识别")
window.geometry("500x600")

# 标题标签
label_title = tk.Label(window, text="请选择图片进行识别", font=("微软雅黑", 16))
label_title.pack(pady=10)

# 图片显示区域
canvas = tk.Canvas(window, width=300, height=300)
canvas.pack()

# 显示预测结果标签
label_result = tk.Label(window, text="", font=("微软雅黑", 14))
label_result.pack(pady=10)

# 保存当前图片路径
current_img_path = None

def select_image():
    global current_img_path
    file_path = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg *.png *.jpeg")])
    if not file_path:
        return
    current_img_path = file_path

    # 加载并显示图片
    image = Image.open(file_path)
    image = image.resize((300, 300))
    tk_img = ImageTk.PhotoImage(image)
    canvas.image = tk_img
    canvas.create_image(0, 0, anchor='nw', image=tk_img)

    # 清空预测结果
    label_result.config(text="")

def predict():
    if not current_img_path:
        label_result.config(text="请先选择图片")
        return

    # 预测图片
    image = Image.open(current_img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        result = classes[predicted.item()]
    if(result=="Dog"):
        label_result.config(text=f"识别结果: 大狗嚼")
    else:
        label_result.config(text=f"识别结果: 哈基米")

# 按钮区域
frame_btns = tk.Frame(window)
frame_btns.pack(pady=10)

btn_select = tk.Button(frame_btns, text="选择图片", command=select_image, width=15)
btn_select.grid(row=0, column=0, padx=10)

btn_predict = tk.Button(frame_btns, text="识别", command=predict, width=15)
btn_predict.grid(row=0, column=1, padx=10)

# 启动 GUI
window.mainloop()
