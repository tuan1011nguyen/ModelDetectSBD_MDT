import cv2
from matplotlib import pyplot as plt
import numpy as np
import onnxruntime as ort
import time
from datasetProcessing import generate_consecutive_pairs


def load_onnx_model(model_path):
    """
    Tải mô hình ONNX.
    
    Parameters:
        - model_path: Đường dẫn tới mô hình ONNX.
    
    Returns:
        - session: Phiên làm việc của mô hình.
        - input_name: Tên input của mô hình.
    """
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    return session, input_name


# Hàm cắt ảnh
def crop_image(image, row_step=23, col_step=16, row_start=0, row_end=208, col_start=0, col_end=32, crop_height=23, crop_width=32):
    crop_img_sbd_mdt = []
    for i in range(row_start, row_end, row_step):
        for j in range(col_start, col_end, col_step):
            imgcrop = image[i:i+crop_height, j:j+crop_width]
            crop_img_sbd_mdt.append(imgcrop)
    return crop_img_sbd_mdt


image = cv2.imread('/mnt/d/examgrading/BuilModelDetecSBD_DT/Dataset/arena/mdt/MDT.c2_quanghung.0006.jpg')
# Sửa thông số và gọi hàm crop_image
list_crop = crop_image(image)

print(len(list_crop))
for image in list_crop:
    plt.imshow(image)
    plt.show()