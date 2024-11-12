import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
from random import choice
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import cv2
import time
from sklearn.metrics import confusion_matrix
class CustomImageDataset(Dataset):
    def __init__(self, fpath, categories, transform=None):
        self.fpath = fpath
        self.categories = categories
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_category = {}
        
        # Duyệt qua từng thư mục và lưu trữ đường dẫn ảnh cùng với nhãn
        for index, category in enumerate(categories):
            category_path = os.path.join(fpath, category)
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                self.image_paths.append(image_path)
                self.labels.append(index)
            self.label_to_category[index] = category
    def __len__(self):
        # Trả về tổng số lượng ảnh
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Lấy đường dẫn ảnh và nhãn tương ứng
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Mở ảnh và chuyển sang RGB
        image = Image.open(image_path).convert('RGB')
        
        # Áp dụng các phép biến đổi nếu có
        if self.transform:
            image = self.transform(image)
        else:
            
            image = image.resize((32, 32))
            image = transforms.ToTensor()(image)
        
        return image, label
    def get_label_mapping(self):
        return self.label_to_category


fpath = "./Dataset/data_bubble"
categories = os.listdir(fpath)
transform = transforms.Compose([
    transforms.Resize((32, 32)),     
    transforms.ToTensor(),        
])
dataset = CustomImageDataset(fpath, categories, transform=transform)
total_size = len(dataset)
train_size = int(0.6 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

label_mapping = dataset.get_label_mapping()
print("Label Mapping:")
for label, category in label_mapping.items():
    print(f"Label {label}: {category}")

import torch
import torch.nn as nn
import torch.optim as optim
class ImprovedCNN(nn.Module):
    def __init__(self, num_categories):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_categories)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    model = model.to(device)
    val_min = 10000.0
    model_save = model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        train_total = 0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        model.eval()
        val_loss = 0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        if (val_loss < val_min):
            model_save = model
            val_min = val_loss
        print(f'Epoch {epoch+1}/{epochs}:', f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%', f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    return model_save
def test_model(model, test_loader, device):
    model = model.to(device)
    model.eval()
    test_total = 0
    test_correct = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_accuracy = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)
def predict_image(model, image_path, transform, device, label_mapping):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    
    predicted_class = predicted.item()
    predicted_class_name = f"Class {label_mapping[predicted_class]}"
    return predicted_class_name
def are_approx_equal(a, b, epsilon=1e-9):
    return abs(a - b) < epsilon
def predict_SBD(model, image_path, transform, device, label_mapping, cols_c):
    model = model.to(device)
    model.eval()
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn!")
    else:
    # Kích thước ảnh (cao, rộng, kênh)
        img_height, img_width, _ = img.shape
        cols = cols_c
        rows = 10
        tile_width = img_width // cols
        tile_height = img_height // rows
        SBD = []
        for col in range(cols):
            check_row_only_value = 0
            max_softmax = -10000
            for row in range(rows):
                left = col * tile_width
                top = row * tile_height
                right = left + tile_width
                bottom = top + tile_height
                img_crop = img[top:bottom, left:right]
                img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                # plt.imshow(img_rgb)
                # plt.axis('off')  # Turn off axis labels
                # plt.show()
                img_pil = Image.fromarray(img_rgb)
                image = transform(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(image)
                    softmax, predicted = outputs.max(1)
                    predicted_class = predicted.item()
                    # print(col + 1, row + 1, f"Class {label_mapping[predicted_class]}")
                    if (check_row_only_value == 0 and predicted == 0):
                        SBD.append(row)
                        max_softmax = softmax
                        check_row_only_value = 1
                    # elif (predicted == 0 and softmax > max_softmax):
                    #     SBD[col] = row
                    #     max_softmax = softmax
                    elif (predicted == 0):
                        print(max_softmax.item() - softmax.item())
                        if (are_approx_equal(softmax, max_softmax)):
                            SBD[col] = 'X'
                            break
                        elif (softmax > max_softmax):
                            SBD[col] = row
                            max_softmax = softmax
                        # print(softmax, max_softmax)
            # print(check_row_only_value)
            if (check_row_only_value == 0):
                SBD.append('_')
        for i in range (cols):
            print(SBD[i], end="")
def predict_SBD_v2(model, image_path, transform, device, label_mapping):
    model = model.to(device)
    model.eval()
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn!")
    else:
    # Kích thước ảnh (cao, rộng, kênh)
        img_height, img_width, _ = img.shape
        cols = 6
        rows = 10
        tile_width = img_width // cols
        tile_height = img_height // rows
        SBD = []
        tiles = []
        for col in range(cols):
            check_row_only_value = 0
            max_softmax = -10000
            for row in range(rows):
                left = col * tile_width
                top = row * tile_height
                right = left + tile_width
                bottom = top + tile_height
                img_crop = img[top:bottom, left:right]
                img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                # plt.imshow(img_rgb)
                # plt.axis('off')  # Turn off axis labels
                # plt.show()
                img_pil = Image.fromarray(img_rgb)
                image = transform(img_pil).unsqueeze(0).to(device)
                tiles.append(image)
                # with torch.no_grad():
                #     outputs = model(image)
                #     softmax, predicted = outputs.max(1)
                #     predicted_class = predicted.item()
                    # print(col + 1, row + 1, f"Class {label_mapping[predicted_class]}")
        tiles_batch =  torch.cat(tiles)
        with torch.no_grad():
            outputs = model(tiles_batch)
            softmax, predicted = outputs.max(1)
            indices = (predicted == 0).nonzero(as_tuple=True)[0]
        #         if (len(indices) == 0):
        #             SBD.append("_")
        #         elif (len(indices) == 1):
        #             SBD.append(indices[0])
        #         elif (len(indices) > 1):
        #             check = -1
        #             for i in range(len(indices)):
        #                 if (are_approx_equal(max_softmax, softmax[i])):
        #                     SBD[-1] = 'X'
        #                     max_softmax = softmax[i]
        #                 elif (check == -1):
        #                     SBD.append(indices[i])
        #                     check = 1
        #                     max_softmax = softmax[i]
        #                 else:
        #                     SBD[-1] = indices[i]
        #                     max_softmax = softmax[i]
        # for i in range(len(SBD)):
        #     print(SBD[i].item(), end="")
def main():
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = ImprovedCNN(num_classes)
    # model = train_model(model, train_loader, val_loader, 20, 0.001, device)
    # test_model(model, test_loader, device)
    # torch.save(model, 'model_CNN_2.pth')
    model = torch.load(r'./models//model_CNN_32x32.pth').to(device)
    # model = nn.DataParallel(model)
    time_start = time.time()
    # print(predict_image(model, r"unmark1/0.0.0.0.5.c3dtnttienyen_phamminhcuong.0002.jpg", transform, device, label_mapping))
    predict_SBD(model, r"Dataset/arena/sbd/SBD.nguyenvanchien1208.0053.jpg", transform, device, label_mapping, 6)
    print("\n")
    predict_SBD(model, r"Dataset/arena/mdt/MDT.khiemphd.0001.jpg", transform, device, label_mapping, 3)
    print('\n')
    predict_SBD_v2(model, r"Dataset/testImage.png", transform, device, label_mapping)
    
    
    time_run = time.time() - time_start
    print("\ntime run: ", time_run/60)

import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()