import ast
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import numpy as np

# # Đọc dữ liệu từ file CSV và chuyển đổi nhãn
# data = pd.read_csv('./data/cropped_image_results.csv')
# label_mapping = {'(0, 0)': 0, '(0, 1)':1,'(1, 0)':2,'(1, 1)': 3}  # Define your mapping
# data['result'] = data['result'].map(label_mapping)
# labels = data['result'].values
# print(labels)
# # print(f"Is CUDA Available: {tf.config.list_physical_devices('GPU')}")

# # Chia dữ liệu thành tập huấn luyện và kiểm tra
# train_paths, test_paths, train_labels, test_labels = train_test_split(
#     data['image_path'], labels, test_size=0.2, random_state=42
# )

# # Định nghĩa phương thức để đọc và xử lý ảnh
# # Định nghĩa phương thức để đọc và xử lý ảnh
# def process_image(img_path, target_size=(32, 23)):
#     img = image.load_img(img_path, target_size=target_size)
#     img = image.img_to_array(img)  # Chuyển ảnh thành numpy array
#     img = img / 255.0  # Chuẩn hóa ảnh
#     return img

# # Chuẩn bị dữ liệu huấn luyện và kiểm tra
# train_images = np.array([process_image(img_path) for img_path in train_paths])
# test_images = np.array([process_image(img_path) for img_path in test_paths])

# train_images = np.array(train_images)
# test_images = np.array(test_images)
# train_labels = np.array(train_labels)
# test_labels = np.array(test_labels)



train_images = np.load('/mnt/d/examgrading/BuilModelDetecSBD_DT/DataNumpy/train_images.npy')
train_labels = np.load('/mnt/d/examgrading/BuilModelDetecSBD_DT/DataNumpy/train_labels.npy')
test_images = np.load('/mnt/d/examgrading/BuilModelDetecSBD_DT/DataNumpy/test_images.npy')
test_labels = np.load('/mnt/d/examgrading/BuilModelDetecSBD_DT/DataNumpy/test_labels.npy')


model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 23, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 2 lớp cho phân loại nhị phân
])

# Compile mô hình
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
num_epochs = 10
batch_size = 4
print('Bắt đầu quá trình train model')
history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(test_images, test_labels))

# Đánh giá mô hình trên tập kiểm tra
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Lưu mô hình
model.save('cnn_model.h5')
print("Model saved successfully.")
