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


def crop_image(image, row_step=23, col_step=16, row_start=0, row_end=208, col_start=0, col_end=65, crop_height=23, crop_width=32):
    """
    Cắt ảnh thành các mảnh con theo cấu trúc bước nhảy.
    
    Parameters:
        - image: Ảnh đầu vào (numpy array).
        - row_step: Bước nhảy theo chiều dọc.
        - col_step: Bước nhảy theo chiều ngang.
        - row_start, row_end: Vị trí bắt đầu và kết thúc cắt theo chiều dọc.
        - col_start, col_end: Vị trí bắt đầu và kết thúc cắt theo chiều ngang.
        - crop_height, crop_width: Kích thước của mỗi mảnh ảnh cắt ra.
    
    Returns:
        - Danh sách các mảnh ảnh con.
    """
    crop_img_sbd_mdt = []
    for i in range(row_start, row_end, row_step):
        for j in range(col_start, col_end, col_step):
            imgcrop = image[i:i+crop_height, j:j+crop_width]
            crop_img_sbd_mdt.append(imgcrop)
    return crop_img_sbd_mdt


def preprocess_image(image, target_size=(32, 23)):
    """
    Tiền xử lý ảnh: thay đổi kích thước và chuẩn hóa giá trị pixel.
    
    Parameters:
        - image: Ảnh đầu vào (numpy array).
        - target_size: Kích thước mục tiêu (width, height).
    
    Returns:
        - image_array: Mảng NumPy của ảnh đã được tiền xử lý.
    """
    # Thay đổi kích thước ảnh
    image = cv2.resize(image, target_size[::-1])

    # Chuyển ảnh thành mảng NumPy và chuẩn hóa giá trị pixel (0-255 -> 0-1)
    image_array = np.array(image).astype(np.float32) / 255.0

    # Kiểm tra xem ảnh có 3 kênh (RGB) hay không
    if image_array.shape[-1] == 3:
        # Chuyển ảnh từ (height, width, channels) thành (channels, height, width)
        image_array = np.transpose(image_array, (2, 0, 1))  # Tạo dạng (C, H, W)
    else:
        raise ValueError("Ảnh không phải RGB hoặc có số kênh khác 3")

    # Thêm chiều batch, ảnh đầu vào sẽ có shape (1, 3, 32, 23)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.transpose(image_array, (0, 2, 3, 1))  # Chuyển về (batch, height, width, channels)
    
    return image_array


def classify_image(session, input_name, image_array):
    """
    Dự đoán lớp cho ảnh đầu vào.
    
    Parameters:
        - session: Phiên làm việc của mô hình ONNX.
        - input_name: Tên input của mô hình.
        - image_array: Mảng NumPy của ảnh đã được tiền xử lý.
    
    Returns:
        - predicted_class: Lớp dự đoán của ảnh.
        - outputs: Các đầu ra từ mô hình.
    """
    outputs = session.run(None, {input_name: image_array})
    predicted_class = np.argmax(outputs[0], axis=1)  # Lấy lớp có xác suất cao nhất
    return predicted_class[0], outputs


def reverse_map_values_to_indices(mapped_tuples, original_tuples):
    """
    Hàm đảo ngược để khôi phục chỉ số từ đầu ra của map_values_to_pairs.
    
    Parameters:
        - mapped_tuples: Các cặp giá trị đã được ánh xạ.
        - original_tuples: Các cặp giá trị gốc.
    
    Returns:
        - indices: Danh sách các chỉ số được khôi phục.
    """
    indices = set()
    for i, (x, y) in enumerate(mapped_tuples):
        if (x, y) == (1, 0):
            indices.add(original_tuples[i][0])
        elif (x, y) == (0, 1):
            indices.add(original_tuples[i][1])
        elif (x, y) == (1, 1):
            indices.update(original_tuples[i])
    return sorted(indices)


def process_image_and_classify(image_path, model_path):
    """
    Xử lý ảnh, phân loại và tính toán kết quả.
    
    Parameters:
        - image_path: Đường dẫn tới ảnh cần xử lý.
        - model_path: Đường dẫn tới mô hình ONNX.
        - label_check: Từ điển ánh xạ kết quả dự đoán thành nhãn.
        - dict_number: Từ điển ánh xạ các chỉ số kết quả.
    
    Returns:
        - result_string: Chuỗi kết quả.
        - total_time: Thời gian tổng cộng cho dự đoán.
    """
    # Tải mô hình ONNX
    session, input_name = load_onnx_model(model_path)

    # Đọc ảnh và cắt thành các mảnh
    image_raw = cv2.imread(image_path)
    list_crop = crop_image(image_raw)

    # Lưu trữ thời gian bắt đầu
    start_time = time.time()


    label_check = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    dict_number = {i: i // 6 for i in range(60)}

    # Tạo từ điển để lưu kết quả
    results = []

    # Lặp qua từng ảnh trong danh sách
    for idx, image in enumerate(list_crop):
        start = time.time()

        # Tiền xử lý ảnh
        image_array = preprocess_image(image)

        # Dự đoán lớp ảnh
        predicted_class, outputs = classify_image(session, input_name, image_array)
        
        # Lưu kết quả vào danh sách
        result = {
            "image_index": idx,
            "Output": outputs,
            "predicted_class_index": predicted_class,
            "predicted_class_label": label_check[predicted_class],
            "prediction_time": time.time() - start
        }
        results.append(result)

    # Khôi phục chỉ số từ kết quả phân loại
    tuples_result = [res['predicted_class_label'] for res in results]
    tuples = generate_consecutive_pairs(list(range(61)))

    # Khôi phục lại các chỉ số
    recovered_indices = reverse_map_values_to_indices(tuples_result, tuples)

    # Sắp xếp mảng theo các nhóm check_indicesAxis1 đến check_indicesAxis6
    check_indices = [set(range(i, 60, 6)) for i in range(6)]
    sorted_array = sorted(recovered_indices, key=lambda x: tuple(x in idx for idx in check_indices), reverse=True)

    # Tính toán giá trị tương ứng với các chỉ số
    values_for_recovered_indices = [dict_number[key] for key in sorted_array]
    result_string = ''.join(map(str, values_for_recovered_indices))

    # Thời gian dự đoán tổng
    total_time = time.time() - start_time
    return result_string, total_time
