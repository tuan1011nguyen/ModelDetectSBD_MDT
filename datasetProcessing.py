import os
import cv2
import pandas as pd
import numpy as np

# Hàm cắt ảnh
def crop_image(image, row_step=23, col_step=16, row_start=0, row_end=208, col_start=0, col_end=65, crop_height=23, crop_width=32):
    crop_img_sbd_mdt = []
    for i in range(row_start, row_end, row_step):
        for j in range(col_start, col_end, col_step):
            imgcrop = image[i:i+crop_height, j:j+crop_width]
            crop_img_sbd_mdt.append(imgcrop)
    return crop_img_sbd_mdt

# Hàm tính toán chỉ số 
def calculate_indices(input_string):
    indeces = []
    for i in range(len(input_string)):
        index_label = i + 6 * int(input_string[i])
        indeces.append(index_label)
    return indeces

# Hàm tạo các cặp liên tiếp
def generate_consecutive_pairs(arr):
    result = []
    append_count = 0
    for i in range(len(arr) - 1):
        if append_count == 5:
            append_count = 0
            continue
        result.append((arr[i], arr[i + 1]))
        append_count += 1
    return result

# Hàm thay thế giá trị trong các cặp
def map_values_to_pairs(tuples, input_values):
    result = []
    for t in tuples:
        updated_tuple = tuple(1 if x in input_values else 0 for x in t)
        result.append(updated_tuple)
    return result






# # Đọc dữ liệu từ Excel
# data = pd.read_excel('processed_files.xlsx')
# # checkSbd = data['checkSBD']
# checkSbd = data['checkSBD'].apply(lambda x: str(x).zfill(6))
# filename = data['Filename']

# # Đọc danh sách ảnh
# folder_path = './Dataset/arena/sbd'
# image_paths = []
# for filename in os.listdir(folder_path):
#     if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
#         image_paths.append(os.path.join(folder_path, filename))

# # Tạo các cặp liên tiếp
# arr = list(range(61))
# tuples = generate_consecutive_pairs(arr)

# # Tạo thư mục 'data' nếu chưa có
# output_image_folder = './data'
# os.makedirs(output_image_folder, exist_ok=True)

# # Tạo danh sách để lưu kết quả
# results = []

# # Duyệt qua tất cả ảnh
# for idx, image_path in enumerate(image_paths):
#     checkSbd_in_image = checkSbd[idx]
#     print(checkSbd_in_image)
#     image = cv2.imread(image_path)
    
#     # Tính toán chỉ số
#     indeces = calculate_indices(checkSbd_in_image)
    
#     # Áp dụng các cặp liên tiếp
#     result = map_values_to_pairs(tuples, indeces)
    
#     # Cắt ảnh thành các mảnh nhỏ
#     list_crop = crop_image(image)
    
#     # Lưu ảnh và ghi kết quả vào file CSV
#     for jdx, image_crop in enumerate(list_crop):
#         # Tạo tên ảnh cho mảnh cắt
#         cropped_image_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{jdx}.png"

#         cropped_image_path = os.path.join(output_image_folder, cropped_image_name)
        
#         # Lưu ảnh cắt ra
#         if cv2.imwrite(cropped_image_path, image_crop):
#             print(f"Saved: {cropped_image_name}")
#         else:
#             print(f"Failed to save: {cropped_image_name}")

        
#         # Thêm kết quả vào danh sách
#         results.append([cropped_image_path, result[jdx]])

# # Lưu kết quả vào file CSV
# results_df = pd.DataFrame(results, columns=['image_path', 'result'])
# results_df.to_csv('./data/cropped_image_results.csv', index=False)
