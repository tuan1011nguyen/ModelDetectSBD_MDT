import cv2
from matplotlib import pyplot as plt

from preprocessingSBD import process_image_and_classify


# Sử dụng hàm:
if __name__ == "__main__":
    image_path = '/mnt/d/examgrading/BuilModelDetecSBD_DT/Dataset/arena/sbd/SBD.c3hoabinhlatrobehn.0017.jpg'
    model_path = '/mnt/d/examgrading/BuilModelDetecSBD_DT/model/cnn_model.onnx'

    result_string, total_time = process_image_and_classify(image_path, model_path)

    print(f"Số báo danh: {result_string}")
    print(f"Thời gian dự đoán tổng: {total_time} giây")
    image = cv2.imread(image_path)
    plt.imshow(image)
    plt.show()
