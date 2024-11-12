
# import time
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np

# # Tải mô hình đã lưu
# model = tf.keras.models.load_model('cnn_model.h5')

# # Hàm dự đoán nhãn của một ảnh
# def predict_image(image_path):
#     start_time = time.time()
#     # Mở ảnh và tiền xử lý
#     img = image.load_img(image_path, target_size=(32, 23))  # Đảm bảo kích thước ảnh giống như khi huấn luyện
#     img_array = image.img_to_array(img)  # Chuyển ảnh thành mảng numpy
#     img_array = np.expand_dims(img_array, axis=0)  # Thêm một dimension để tạo ra batch size 1

#     # Tiền xử lý ảnh (nếu có, ví dụ như chuẩn hóa hoặc chuyển đổi)
#     img_array = img_array / 255.0  # Đưa giá trị ảnh vào khoảng [0, 1] nếu mô hình yêu cầu

#     # Dự đoán nhãn
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions, axis=1)  # Lấy nhãn với xác suất cao nhất
#     end_time = time.time()
#     elapsed_time = end_time - start_time

#     print(f"Thời gian chạy: {elapsed_time:.4f} seconds")
#     confidence = np.max(predictions)  # Lấy độ tin cậy của dự đoán
#     return predicted_class[0], confidence

# # Đường dẫn đến ảnh bạn muốn dự đoán
# image_path = "/mnt/d/examgrading/BuilModelDetecSBD_DT/data/SBD.c2_quanghung.0001_0.png"
# predicted_class, confidence = predict_image(image_path)

# print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}')
# import onnx
# import tensorflow as tf
# import tf2onnx


# # Load the Keras model
# model = tf.keras.models.load_model('/mnt/d/examgrading/BuilModelDetecSBD_DT/model/cnn_model.h5')

# # Wrap the Sequential model in a Functional model
# functional_model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs)

# # Define the input signature
# input_signature = [tf.TensorSpec(shape=(None, 32, 23, 3), dtype=tf.float32)]  # Adjust input shape as necessary

# # Convert the Functional model to ONNX
# onnx_model, _ = tf2onnx.convert.from_keras(functional_model, input_signature=input_signature, opset=13)

# # Save the ONNX model
# onnx.save(onnx_model, "/mnt/d/examgrading/BuilModelDetecSBD_DT/model/cnn_model.onnx")
from PIL import Image
import numpy as np
import onnxruntime as ort
import time

# Đường dẫn tới mô hình ONNX
onnx_model_path = "/mnt/d/examgrading/BuilModelDetecSBD_DT/model/cnn_model.onnx"

# Tải mô hình ONNX
session = ort.InferenceSession(onnx_model_path)

# Lấy tên input của mô hình
input_name = session.get_inputs()[0].name

# Danh sách các đường dẫn tới ảnh
image_paths = [
    "/mnt/d/examgrading/BuilModelDetecSBD_DT/Dataset/cropped_images/SBD_crop_5.png",
    "/mnt/d/examgrading/BuilModelDetecSBD_DT/Dataset/cropped_images/SBD_crop_6.png",
    # Thêm các đường dẫn ảnh khác ở đây
]

# Tạo từ điển để lưu kết quả
results = []
label_check = {0: "(0, 0)", 1: "(0, 1)", 2: "(1, 0)", 3: "(1, 1)"}

# Lặp qua từng ảnh trong danh sách
for image_path in image_paths:
    # Bắt đầu tính thời gian
    start = time.time()

    # Tải ảnh và thay đổi kích thước thành (23, 32)
    image = Image.open(image_path)
    image = image.resize((23, 32))  # Kích thước (23, 32)

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

    # Chuyển đổi lại thứ tự các chiều từ (batch_size, channels, height, width) thành (batch_size, height, width, channels)
    image_array = np.transpose(image_array, (0, 2, 3, 1))  # Chuyển từ (1, 3, 32, 23) thành (1, 32, 23, 3)

    # Chạy mô hình và lấy kết quả
    outputs = session.run(None, {input_name: image_array})
    end_time = time.time()
    
    # Giả sử outputs[0] chứa kết quả phân loại
    predicted_class = np.argmax(outputs[0], axis=1)  # Lấy lớp có xác suất cao nhất
    
    # Lưu kết quả vào danh sách
    result = {
        "image_path": image_path,
        "predicted_class_index": predicted_class[0],
        "predicted_class_label": label_check[predicted_class[0]],
        "prediction_time": end_time - start
    }
    results.append(result)

# In ra kết quả cho tất cả các ảnh
for res in results:
    print(f"Image: {res['image_path']}")
    print(f"Predicted class index: {res['predicted_class_index']}")
    print(f"Predicted class label: {res['predicted_class_label']}")
    print(f"Prediction time: {res['prediction_time']} seconds")
    print("="*10)




# import tensorrt as trt
# from logzero import logger
# import yaml
# import os
# import argparse

# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
# EXPLICIT_BATCH |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# def build_engine(model_path, quantize="fp16"):
#     with trt.Builder(TRT_LOGGER) as builder, \
#             builder.create_network(EXPLICIT_BATCH) as network, \
#             builder.create_builder_config() as config, \
#             trt.OnnxParser(network, TRT_LOGGER) as parser:
#         config = builder.create_builder_config()
#         config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB

#         serialized_engine = builder.build_serialized_network(network, config)

#         # Handle quantization flag (FP16)
#         if quantize == "fp16":
#             config.flags |= 1 << int(trt.BuilderFlag.FP16)

#         config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
#         config.default_device_type = trt.DeviceType.GPU

#         # Parse the ONNX model
#         with open(model_path, "rb") as f:
#             if not parser.parse(f.read()):
#                 print('ERROR: Failed to parse the ONNX file.')
#                 for error in range(parser.num_errors):
#                     logger.error(parser.get_error(error))

#         logger.info("ONNX parse ended")
        
#         # Define optimization profile
#         profile = builder.create_optimization_profile()
        
#         # Here, replace `input_min_size`, `input_opt_size`, and `input_max_size` with actual values
#         input_name = "args_0"  # Assuming the input name is 'args_0'
#         min_shape = (1, 32, 23, 3)
#         opt_shape = (1, 32, 23, 3)
#         max_shape = (1, 32, 23, 3)

#         profile.set_shape(input_name, min_shape, opt_shape, max_shape)

        
#         network.add_input(input_name, trt.float32, min_shape)
#         profile.set_shape(input_name, min_shape, opt_shape, max_shape)
#         config.add_optimization_profile(profile)

#         logger.debug(f"config = {config}")

#         logger.info("====================== building tensorrt engine... ======================")
#         engine = builder.build_serialized_network(network, config)
#         logger.info("Engine was created successfully")
        
#         # Save engine to file
#         with open("model.trt", 'wb') as f:
#             try:
#                 f.write(bytearray(engine))
#             except:
#                 logger.error("Failed to write the engine to file")
                
#         return engine

# if __name__ == '__main__':
#     onnx_file_path = "/mnt/d/examgrading/BuilModelDetecSBD_DT/model/cnn_model.onnx"  # Specify your model file path
#     build_engine(model_path=onnx_file_path, quantize="fp16")





# import tensorflow as tf
# import tensorflow.experimental.tensorrt as tftrt

# # Tải mô hình TensorFlow (nếu bạn sử dụng mô hình .h5)
# model = tf.keras.models.load_model('cnn_model.h5')

# # Chuyển đổi mô hình TensorFlow sang TensorRT
# converter = tftrt.TrtGraphConverterV2(input_saved_model_dir='saved_model')
# converter.convert()

# # Lưu mô hình đã chuyển đổi
# converter.save('save_trt_model')  # Đảm bảo lưu đúng định dạng saved_model



# import tensorflow as tf
# from tensorflow.python.compiler.tensorrt import trt_convert as trt

# # Load mô hình TensorFlow (ví dụ là mô hình .h5)
# model = tf.keras.models.load_model('cnn_model.h5')

# # Cấu hình chuyển đổi TensorFlow sang TensorRT
# params = trt.DEFAULT_TRT_OPS_GENERATOR_PARAMS
# params.precision_mode = trt.TrtPrecisionMode.FP16  # hoặc FP32 hoặc INT8 nếu muốn
# params.max_workspace_size_bytes = 1 << 30  # 1GB

# # Chuyển đổi mô hình TensorFlow sang TensorRT
# converter = trt.TrtGraphConverterV2(input_saved_model_dir=None, input_saved_model_signature_key=None, input_saved_model_tags=None, input_saved_model=None)
# converter.convert()
# converter.save('model_trt')

# # Hoặc nếu bạn muốn chuyển từ mô hình Keras .h5 sang .trt trực tiếp
# converter = trt.TrtGraphConverterV2(input_saved_model_dir='model_trt')
# converter.convert()
# converter.save('model_trt')


# import cv2
# import numpy as np
# import tensorflow as tf

# # Đọc ảnh test
# img = cv2.imread('/mnt/d/examgrading/BuilModelDetecSBD_DT/Dataset/cropped_images/SBD_crop_1.png')
# img = cv2.resize(img, (32, 23))  # Resize ảnh cho phù hợp với input của mô hình

# # Đảm bảo ảnh có đúng số kênh màu nếu cần
# if img.shape[-1] == 1:  # Nếu ảnh grayscale, chuyển sang 3 kênh
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# # Chuẩn hóa ảnh (nếu mô hình yêu cầu)
# img = img / 255.0  # Chuẩn hóa giá trị pixel về phạm vi [0, 1]

# # Đọc mô hình TensorRT
# model_trt = tf.saved_model.load('/mnt/d/examgrading/BuilModelDetecSBD_DT/model.trt')

# # Dự đoán với mô hình TensorRT
# input_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
# input_tensor = input_tensor[tf.newaxis, ...]  # Thêm batch dimension

# # Kiểm tra mô hình có đầu vào nào cụ thể không
# print("Mô hình TensorRT đầu vào:", model_trt.signatures)

# # Thực hiện dự đoán
# output = model_trt(input_tensor)

# # Xử lý output theo yêu cầu (ví dụ: phân loại, nhận diện...)
# print(output)
