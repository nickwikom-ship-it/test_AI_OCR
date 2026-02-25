import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="AI Multi-Lang OCR",
    page_icon="🔍",
    layout="wide"
)

# แก้ไข Parameter เป็น unsafe_allow_html=True เพื่อรองรับเวอร์ชันล่าสุด
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextArea textarea {
        font-size: 18px !important;
        font-family: 'Tahoma', sans-serif;
    }
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📸 AI Multi-Language Text Scanner")
st.write("ระบบดึงข้อความจากภาพ (รองรับ ไทย, อังกฤษ, จีน) - Optimized for Speed")

# --- ส่วน Sidebar ---
st.sidebar.header("⚙️ การตั้งค่า")

# เลือกภาษา
lang_options = {
    "ไทย + English": "tha+eng",
    "English Only": "eng",
    "ภาษาจีน (ตัวย่อ)": "chi_sim",
    "ภาษาจีน (ตัวเต็ม)": "chi_tra",
    "ไทย + จีน": "tha+chi_sim"
}
selected_option = st.sidebar.selectbox("เลือกภาษาที่แสดงในภาพ:", list(lang_options.keys()))
selected_lang = lang_options[selected_option]

# ตัวเลือกการปรับภาพ
auto_sharpen = st.sidebar.checkbox("เพิ่มความคมชัดอัตโนมัติ (Otsu Threshold)", value=True)

# อัปโหลดไฟล์
uploaded_file = st.sidebar.file_uploader("อัปโหลดรูปภาพ (JPG, PNG)...", type=["jpg", "jpeg", "png"])

# --- ส่วนประมวลผลหลัก ---
if uploaded_file is not None:
    # 1. โหลดภาพ
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ ภาพต้นฉบับ")
        st.image(image, use_container_width=True)

    # 2. กระบวนการ Image Enhancement
    with st.spinner('กำลังประมวลผลภาพ...'):
        # แปลงเป็นขาวดำ
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if auto_sharpen:
            # ใช้ Otsu's Threshold แยกพื้นหลังและตัวอักษร (รวดเร็วและแม่นยำสูง)
            _, processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            processed_img = gray

    with col2:
        st.subheader("✨ ภาพหลังปรับแต่ง")
        st.image(processed_img, use_container_width=True, channels="GRAY")

    # 3. กระบวนการ OCR
    st.divider()
    with st.spinner(f'AI กำลังอ่านข้อความ ({selected_option})...'):
        try:
            # --oem 1: LSTM OCR Engine
            # --psm 6: Assume a single uniform block of text
            custom_config = r'--oem 1 --psm 6'
            
            # รัน OCR ผ่าน Tesseract
            text = pytesseract.image_to_string(processed_img, lang=selected_lang, config=custom_config)
            
            if text.strip():
                st.subheader("📄 ข้อความที่สแกนได้:")
                # แสดงผลใน Text Area เพื่อให้ก๊อปปี้ง่าย
                st.text_area("Result", text, height=400, label_visibility="hidden")
                
                # ปุ่มดาวน์โหลด
                st.download_button(
                    label="📥 ดาวน์โหลดไฟล์ข้อความ (.txt)",
                    data=text.encode('utf-8'),
                    file_name="scanned_result.txt",
                    mime="text/plain"
                )
            else:
                st.warning("⚠️ AI ไม่พบข้อความในรูปภาพ โปรดตรวจสอบการเลือกภาษาหรือคุณภาพของรูปภาพ")
        
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
            st.info("ตรวจสอบว่าไฟล์ 'packages.txt' มีการลง tesseract-ocr-tha/eng/chi ครบถ้วน")

else:
    # หน้าจอแรกรับ
    st.info("💡 คำแนะนำ: อัปโหลดรูปภาพจากแถบด้านข้าง (Sidebar) เพื่อเริ่มสแกนข้อความ")
    
    # คำแนะนำการใช้งาน
    with st.expander("วิธีใช้งานเบื้องต้น"):
        st.write("""
        1. **เตรียมรูป:** ใช้รูปที่ตัวอักษรไม่ซ้อนทับกันมากเกินไป
        2. **เลือกภาษา:** สำคัญมาก หากรูปมีภาษาจีนแต่เลือก Thai+Eng AI จะอ่านเพี้ยน
        3. **ดาวน์โหลด:** เมื่อสแกนเสร็จ สามารถกดปุ่มดาวน์โหลดไฟล์ไปใช้งานต่อได้ทันที
        """)
