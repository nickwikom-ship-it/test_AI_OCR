import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Multi-Lang OCR", layout="wide")

st.title("📸 AI Multi-Language Image Enhancer & OCR")
st.write("เลือกภาษา ปรับรูปภาพให้ชัด และสแกนข้อความด้วย AI")

# --- ส่วน Sidebar สำหรับตั้งค่า ---
st.sidebar.header("⚙️ การตั้งค่า")

# 1. ตัวเลือกภาษา
lang_option = st.sidebar.selectbox(
    "เลือกภาษาที่อยู่ในรูปภาพ:",
    ("Thai + English", "English Only", "Chinese (Simplified)", "Chinese (Traditional)", "Thai + Chinese")
)

# แมพตัวเลือกกับรหัสภาษาของ Tesseract
lang_map = {
    "Thai + English": "tha+eng",
    "English Only": "eng",
    "Chinese (Simplified)": "chi_sim",
    "Chinese (Traditional)": "chi_tra",
    "Thai + Chinese": "tha+chi_sim"
}
selected_lang = lang_map[lang_option]

# 2. ปุ่มอัปโหลดไฟล์
uploaded_file = st.sidebar.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # อ่านภาพ
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ ภาพต้นฉบับ")
        st.image(image, use_container_width=True)

    # --- กระบวนการปรับภาพให้ชัด (Image Enhancement) ---
    with st.spinner('กำลังปรับแต่งความคมชัด...'):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # ขยายภาพ 2 เท่า (Upscaling)
        height, width = gray.shape
        enlarged = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        
        # เพิ่มความคม (Sharpening)
        gaussian_blur = cv2.GaussianBlur(enlarged, (0, 0), 3)
        sharpened = cv2.addWeighted(enlarged, 1.5, gaussian_blur, -0.5, 0)
        
        # Adaptive Threshold (สู้เงาและรอยเปื้อน)
        processed_img = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

    with col2:
        st.subheader("✨ ภาพหลังปรับแต่ง (AI Enhanced)")
        st.image(processed_img, use_container_width=True)

    # --- กระบวนการสแกน OCR ---
    st.divider()
    st.subheader(f"📄 ผลการสแกนภาษา: {lang_option}")
    
    with st.spinner(f'กำลังอ่านข้อความภาษา {lang_option}...'):
        # ตรวจสอบตัวอักษร
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_img, lang=selected_lang, config=custom_config)

    if text.strip():
        st.text_area("ข้อความที่ตรวจพบ:", text, height=350)
        
        # ปุ่มดาวน์โหลดผลลัพธ์
        st.download_button(
            label="📥 ดาวน์โหลดไฟล์ข้อความ (.txt)",
            data=text.encode('utf-8'),
            file_name="ocr_result.txt",
            mime="text/plain"
        )
    else:
        st.warning("⚠️ AI ไม่พบข้อความในรูปภาพ ลองเลือกภาษาให้ตรงกับในรูป หรือใช้รูปที่เห็นตัวอักษรชัดกว่านี้")

else:
    st.info("💡 คำแนะนำ: อัปโหลดรูปภาพที่แถบด้านข้างเพื่อเริ่มการทำงาน")
