import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Text Scanner", layout="wide")

st.title("📸 AI Image Enhancer & OCR Scanner")
st.write("อัปโหลดภาพที่เบลอหรือมองไม่ชัด เพื่อให้ AI ปรับแต่งและสแกนข้อความ (รองรับ ไทย-อังกฤษ)")

# ส่วนการอัปโหลดไฟล์
uploaded_file = st.sidebar.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แปลงไฟล์ที่อัปโหลดเป็นภาพ OpenCV
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ต้นฉบับ")
        st.image(image, use_container_width=True)

    # --- ขั้นตอนการ Image Processing เพื่อเพิ่มความชัด ---
    # 1. แปลงเป็นขาวดำ
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2. ขยายขนาดภาพ (Upscaling) เพื่อให้ตัวอักษรชัดขึ้น
    height, width = gray.shape
    enlarged = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    
    # 3. เพิ่มความคม (Sharpening)
    gaussian_blur = cv2.GaussianBlur(enlarged, (0, 0), 3)
    sharpened = cv2.addWeighted(enlarged, 1.5, gaussian_blur, -0.5, 0)
    
    # 4. ทำ Adaptive Thresholding (สู้กับแสงเงา)
    processed_img = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    with col2:
        st.subheader("ภาพที่ AI ปรับแต่งแล้ว")
        st.image(processed_img, use_container_width=True)

    # --- ขั้นตอนการสแกน OCR ---
    st.divider()
    with st.spinner('กำลังสแกนข้อความ...'):
        # กำหนดภาษาเป็น ไทย และ อังกฤษ
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_img, lang='tha+eng', config=custom_config)

    st.subheader("📄 ข้อความที่ตรวจพบ:")
    if text.strip():
        st.text_area("Result", text, height=300)
        st.download_button("ดาวน์โหลดข้อความ (.txt)", text, file_name="scanned_text.txt")
    else:
        st.warning("ไม่พบข้อความในรูปภาพ ลองปรับแสงหรือใช้รูปที่ชัดขึ้น")