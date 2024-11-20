import streamlit as st
import torch
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import timm
from collections import OrderedDict
from torchvision import transforms
from ultralytics import YOLO

# Định nghĩa các biến đổi ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Tải mô hình YOLO
try:
    model_detect = YOLO('best.pt')  # Đảm bảo best.pt tồn tại
except Exception as e:
    st.error(f"Lỗi tải mô hình YOLO: {e}")
    st.stop()

# Tải mô hình phân loại Xception
try:
    model_classify = timm.create_model('xception', pretrained=False, num_classes=4)
    state_dict = torch.load('XceptionNet_chicken_disease.pt', map_location='cpu')  # Đảm bảo file tồn tại
    # Xử lý state_dict nếu có module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        name = name.replace('fc.1', 'fc')
        new_state_dict[name] = v
    model_classify.load_state_dict(new_state_dict, strict=False)
    model_classify.eval()  # Chuyển mô hình sang chế độ đánh giá
except Exception as e:
    st.error(f"Lỗi tải mô hình phân loại: {e}")
    st.stop()

# Labels cho các loại bệnh
labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# Tạo giao diện Streamlit
st.title("Phát hiện bệnh ở phân gà")
uploaded_file = st.file_uploader("Tải lên ảnh phân gà", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # Load ảnh
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Ảnh gốc', use_column_width=True)

        # Dự đoán vùng phát hiện bệnh bằng YOLO
        results = model_detect(image)
        if len(results[0].boxes) == 0:
            st.warning("Không phát hiện được vùng nghi ngờ trong ảnh!")
            st.stop()

        # Lấy bounding box đầu tiên
        xmin, ymin, xmax, ymax = results[0].boxes.xyxy[0].cpu().numpy()

        # Cắt ảnh theo vùng phát hiện
        img_crop = image.crop((xmin, ymin, xmax, ymax))

        # Chuẩn bị ảnh cho mô hình phân loại
        img_tensor = transform(img_crop).unsqueeze(0)

        # Dự đoán loại bệnh
        with torch.no_grad():
            predict = model_classify(img_tensor)
            predicted_label = labels[torch.argmax(predict, dim=1).item()]

        # Vẽ bounding box và nhãn lên ảnh
        draw = ImageDraw.Draw(image)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin - 15), predicted_label, fill="red")

        # Hiển thị kết quả
        st.image(image, caption=f"Phân loại: {predicted_label}", use_column_width=True)

    except Exception as e:
        st.error(f"Lỗi xử lý ảnh: {e}")
