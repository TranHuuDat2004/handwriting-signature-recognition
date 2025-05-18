import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import glob # Để tìm file theo pattern
from sklearn.model_selection import train_test_split # Để chia train/test
from PIL import Image

# --- Cấu hình cho dữ liệu chữ ký ---
BASE_DATA_DIR = '.' # <<=== THAY ĐỔI ĐƯỜNG DẪN NÀY (thư mục cha của full_forg, full_org)
FORG_FOLDER_NAME = 'full_forg'
ORG_FOLDER_NAME = 'full_org'

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
CHANNELS = 1      # Ảnh xám
BATCH_SIZE = 32
VALIDATION_SPLIT_RATIO = 0.2 # Tỷ lệ dữ liệu dùng cho validation
SEED = 42 # Để đảm bảo kết quả lặp lại

# 1. Thu thập đường dẫn file và trích xuất nhãn (ID người viết)
all_image_paths = []
all_writer_ids_str = [] # Lưu ID người viết dưới dạng string trước

folders_to_scan = [os.path.join(BASE_DATA_DIR, FORG_FOLDER_NAME),
                   os.path.join(BASE_DATA_DIR, ORG_FOLDER_NAME)]

print(f"Đang quét các thư mục: {folders_to_scan}")

for folder_path in folders_to_scan:
    if not os.path.isdir(folder_path):
        print(f"CẢNH BÁO: Không tìm thấy thư mục {folder_path}. Bỏ qua.")
        continue
    # Giả sử các file có đuôi .png, bạn có thể thay đổi nếu cần
    # Có thể là '*.jpg', '*.tif', v.v.
    # Bạn có thể thêm nhiều pattern: image_files = glob.glob(os.path.join(folder_path, '*.png')) + glob.glob(os.path.join(folder_path, '*.tif'))
    image_files = glob.glob(os.path.join(folder_path, '*.png'))
    if not image_files:
        image_files = glob.glob(os.path.join(folder_path, '*.tif')) # Thử .tif nếu không có .png
    if not image_files:
        image_files = glob.glob(os.path.join(folder_path, '*.jpg')) # Thử .jpg
    
    print(f"Tìm thấy {len(image_files)} file trong {folder_path}")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        parts = filename.split('_') # Ví dụ: "forgeries_1_2.png" -> ["forgeries", "1", "2.png"]
                                    # Hoặc "genuine_1_2.png" -> ["genuine", "1", "2.png"]
        if len(parts) >= 2: # Cần ít nhất 2 phần để có ID người viết
            writer_id_str = parts[1] # ID người viết là phần tử thứ 2
            all_image_paths.append(img_path)
            all_writer_ids_str.append(writer_id_str)
        else:
            print(f"CẢNH BÁO: Không thể trích xuất ID người viết từ tên file: {filename}")

if not all_image_paths:
    print("LỖI: Không tìm thấy file ảnh nào. Vui lòng kiểm tra lại đường dẫn và cấu trúc thư mục.")
    exit()

print(f"Tổng số ảnh tìm thấy: {len(all_image_paths)}")
print(f"Ví dụ 5 đường dẫn ảnh đầu tiên: {all_image_paths[:5]}")
print(f"Ví dụ 5 ID người viết (string) đầu tiên: {all_writer_ids_str[:5]}")


# 2. Tạo danh sách các lớp (class_names) và chuyển ID người viết thành số nguyên
class_names = sorted(list(set(all_writer_ids_str))) # Danh sách các ID người viết duy nhất, đã sắp xếp
num_classes = len(class_names)

if num_classes == 0:
    print("LỖI: Không có lớp (người viết) nào được tìm thấy. Kiểm tra lại quá trình trích xuất ID.")
    exit()

print(f"Tìm thấy {num_classes} lớp (người viết): {class_names}")

# Tạo mapping từ ID string sang ID số nguyên (0, 1, 2...)
writer_id_to_int_label = {writer_id: i for i, writer_id in enumerate(class_names)}
all_labels_int = [writer_id_to_int_label[writer_id] for writer_id in all_writer_ids_str]

print(f"Ví dụ 5 nhãn số nguyên đầu tiên: {all_labels_int[:5]}")


# 3. Chia dữ liệu thành tập huấn luyện và tập kiểm định
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths,
    all_labels_int,
    test_size=VALIDATION_SPLIT_RATIO,
    random_state=SEED,
    stratify=all_labels_int # Quan trọng để giữ tỷ lệ các lớp trong cả tập train và val
)

print(f"Số lượng mẫu huấn luyện: {len(train_paths)}")
print(f"Số lượng mẫu kiểm định: {len(val_paths)}")


# 4. Tạo hàm để tải và tiền xử lý ảnh
def load_and_preprocess_image(path, label):
    try:
        image_bytes = tf.io.read_file(path)
        # Thử decode nhiều định dạng phổ biến nếu không chắc chắn
        try:
            image = tf.image.decode_png(image_bytes, channels=CHANNELS)
        except tf.errors.InvalidArgumentError:
            try:
                image = tf.image.decode_jpeg(image_bytes, channels=CHANNELS)
            except tf.errors.InvalidArgumentError:
                image = tf.image.decode_image(image_bytes, channels=CHANNELS, expand_animations=False) # Fallback

        image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
        # Đảm bảo ảnh có đúng số kênh
        if image.shape[-1] != CHANNELS and CHANNELS == 1: # Nếu cần ảnh xám mà ảnh đọc vào có 3 kênh
             image = tf.image.rgb_to_grayscale(image)
        elif image.shape[-1] != CHANNELS and CHANNELS == 3 and image.shape[-1] == 1: # Nếu cần ảnh màu mà ảnh đọc vào là xám
             image = tf.image.grayscale_to_rgb(image)

        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
        return image, label
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {path}: {e}")
        # Trả về một tensor rỗng hoặc một giá trị đặc biệt để có thể lọc sau này nếu cần
        # Hoặc raise lỗi để dừng chương trình
        # Tạm thời trả về None để có thể filter
        return None, label


# 5. Tạo tf.data.Dataset
# Huấn luyện
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.filter(lambda x, y: x is not None) # Loại bỏ các mẫu bị lỗi khi tải
train_dataset = train_dataset.shuffle(buffer_size=len(train_paths), seed=SEED)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Kiểm định
val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.filter(lambda x, y: x is not None) # Loại bỏ các mẫu bị lỗi khi tải
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# 6. (Tùy chọn) Xác minh dữ liệu bằng cách vẽ một vài ảnh
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1): # Lấy 1 batch
    for i in range(min(9, images.shape[0])):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray' if CHANNELS == 1 else None)
        plt.title(f"Writer: {class_names[labels[i].numpy()]}")
        plt.axis("off")
plt.suptitle("Một vài mẫu chữ ký từ tập huấn luyện")
plt.show()


# 7. Xây dựng mô hình CNN (Tương tự như trước, đảm bảo input_shape và output units đúng)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Có thể thêm một lớp Conv/Pool nữa nếu cần
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu')) # Có thể tăng units
model.add(layers.Dropout(0.5)) # Dropout giúp chống overfitting
model.add(layers.Dense(num_classes)) # Output layer với số lượng lớp là num_classes

model.summary()

# 8. Compile mô hình
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 9. Huấn luyện mô hình
epochs = 10     # Bạn có thể cần điều chỉnh số epochs
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# 10. Đánh giá mô hình
test_loss, cnn_test_acc = model.evaluate(val_dataset, verbose=2)
print(f"Độ chính xác trên tập kiểm định: {cnn_test_acc:.4f}")

# 11. Vẽ biểu đồ training và validation accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# --- (Tùy chọn) Lưu mô hình ---
# model.save('signature_writer_identification_model.h5')
# print("Mô hình đã được lưu.")

# --- (Tùy chọn) Dự đoán trên ảnh mới ---
# def preprocess_single_image_for_prediction(image_path):
#     # Tải và tiền xử lý tương tự như hàm load_and_preprocess_image nhưng không có label
#     try:
#         image_bytes = tf.io.read_file(image_path)
#         try:
#             image = tf.image.decode_png(image_bytes, channels=CHANNELS)
#         except tf.errors.InvalidArgumentError:
#             try:
#                 image = tf.image.decode_jpeg(image_bytes, channels=CHANNELS)
#             except tf.errors.InvalidArgumentError:
#                 image = tf.image.decode_image(image_bytes, channels=CHANNELS, expand_animations=False)

#         image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
#         if image.shape[-1] != CHANNELS and CHANNELS == 1:
#              image = tf.image.rgb_to_grayscale(image)
#         elif image.shape[-1] != CHANNELS and CHANNELS == 3 and image.shape[-1] == 1:
#              image = tf.image.grayscale_to_rgb(image)
#         image = tf.cast(image, tf.float32) / 255.0
#         return tf.expand_dims(image, 0) # Thêm chiều batch
#     except Exception as e:
#         print(f"Lỗi khi xử lý ảnh {image_path} để dự đoán: {e}")
#         return None

# # Thay 'path_to_new_signature.png' bằng đường dẫn ảnh chữ ký bạn muốn thử
# try:
#     new_signature_path = 'đường_dẫn_đến_ảnh_chữ_ký_mới.png' # <<=== THAY ĐỔI
#     preprocessed_img_for_pred = preprocess_single_image_for_prediction(new_signature_path)
#     if preprocessed_img_for_pred is not None:
#         predictions = model.predict(preprocessed_img_for_pred)
#         score = tf.nn.softmax(predictions[0]) # Áp dụng softmax để có xác suất

#         predicted_class_index = np.argmax(score)
#         predicted_writer_id = class_names[predicted_class_index] # Lấy ID người viết từ class_names
#         confidence = 100 * np.max(score)

#         print(f"Ảnh này được dự đoán là của người viết: {predicted_writer_id} với độ tự tin {confidence:.2f}%")

#         # Hiển thị ảnh
#         img_display = Image.open(new_signature_path)
#         plt.imshow(img_display, cmap='gray' if CHANNELS == 1 else None)
#         plt.title(f"Dự đoán: Người viết {predicted_writer_id} ({confidence:.2f}%)")
#         plt.axis('off')
#         plt.show()
# except FileNotFoundError:
#     print(f"Không tìm thấy ảnh tại: {new_signature_path}")
# except Exception as e:
#     print(f"Lỗi khi dự đoán ảnh mới: {e}")