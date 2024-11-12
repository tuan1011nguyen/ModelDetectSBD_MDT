import ast
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

# # Đọc dữ liệu từ file CSV và chuyển đổi nhãn
# data = pd.read_csv('./data/cropped_image_results.csv')
# label_mapping = {'(0, 0)': 0, '(0, 1)':1,'(1, 0)':2,'(1, 1)': 3}  # Define your mapping
# data['result'] = data['result'].map(label_mapping)
# labels = data['result'].values
# print(labels)
# print(torch.cuda.is_available())

# # Chia dữ liệu thành tập huấn luyện và kiểm tra
# train_paths, test_paths, train_labels, test_labels = train_test_split(
#     data['image_path'], labels, test_size=0.2, random_state=42
# )
# print(f"Unique labels in training data: {np.unique(train_labels)}")
# print(f"Unique labels in test data: {np.unique(test_labels)}")

# Định nghĩa lớp Dataset tùy chỉnh
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.int64)

        if self.transform:
            img = self.transform(img)
        
        return img, label

# Chuyển đổi ảnh và nhãn với Data Augmentation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize ảnh về kích thước 32x32
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# # Tạo Dataset và DataLoader cho tập huấn luyện và kiểm tra
# train_dataset = CustomDataset(train_paths.reset_index(drop=True), train_labels, transform)
# test_dataset = CustomDataset(test_paths.reset_index(drop=True), test_labels, transform)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# # Định nghĩa mô hình CNN
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Lấy kích thước đầu vào của FC1 bằng cách thử nghiệm
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Khởi tạo mô hình, loss function và optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Kiểm tra thiết bị và di chuyển mô hình
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Vòng lặp huấn luyện
# num_epochs = 10
# for epoch in range(num_epochs):
#     print("Start train model")
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
#           f'Accuracy: {100 * correct / total:.2f}%')
#     print(f"Đã train xong lần lặp thứ {epoch}")

# # Đánh giá mô hình trên tập kiểm tra
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Test Accuracy: {100 * correct / total:.2f}%')

# # Lưu trạng thái của mô hình
# torch.save(model.state_dict(), "cnn_model.pth")
# print("Model saved successfully.")
import torch.nn.functional as F
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Khởi tạo lại mô hình và tải trạng thái đã lưu
model = CNNModel()
model.load_state_dict(torch.load("cnn_model.pth"))
model.to(device)
model.eval()  # Chuyển mô hình sang chế độ đánh giá

# Hàm dự đoán nhãn của một ảnh
def predict_image(image_path):
    start_time = time.time()
    # Mở ảnh và thực hiện các bước tiền xử lý
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # Thực hiện chuyển đổi và thêm batch dimension
    img = img.to(device)  # Di chuyển ảnh đến GPU nếu có

    with torch.no_grad():  # Tắt tính toán gradient
        output = model(img)  # Dự đoán nhãn
        probabilities = F.softmax(output, dim=1)  # Tính xác suất
        _, predicted_class = torch.max(output, 1)  # Lấy nhãn dự đoán có xác suất cao nhất
        predicted_class = predicted_class.item()
    end_time = time.time()
    # Tính thời gian dự đoán
    elapsed_time = end_time - start_time

    print(f"Thời gian chạy: {elapsed_time:.4f} seconds")
    return predicted_class, probabilities[0][predicted_class].item()

# Đường dẫn đến ảnh bạn muốn dự đoán
image_path = "/mnt/d/examgrading/BuilModelDetecSBD_DT/Dataset/cropped_images/SBD_crop_2.png"
print('Thiết bị chạy:', device)
predicted_class, confidence = predict_image(image_path)

print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}')
