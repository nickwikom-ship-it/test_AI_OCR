import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

# --- Web Configuration ---
st.set_page_config(page_title="AI Multi-Lang OCR Pro", layout="wide")

# --- UI Language Dictionary ---
# ส่วนนี้คือคลังคำศัพท์สำหรับเปลี่ยนภาษาหน้าเว็บ
ui_strings = {
    "TH": {
        "title": "📸 AI Multi-Language Text Scanner",
        "subtitle": "เวอร์ชันโปร: ลดสัญญาณรบกวนและเชื่อมเส้นตัวอักษรเพื่อความแม่นยำสูง",
        "settings": "⚙️ การตั้งค่าประมวลผล",
        "select_lang_ocr": "เลือกภาษาที่อยู่ในรูป:",
        "noise_label": "ระดับการลบจุดรบกวน",
        "upscale_label": "ขยายขนาดภาพ",
        "upload_label": "อัปโหลดรูปภาพ...",
        "src_img": "🖼️ ภาพต้นฉบับ",
        "proc_img": "✨ ภาพที่ AI ใช้ประมวลผล",
        "scanning": "กำลังแปลงภาพเป็นข้อความ...",
        "cleaning": "AI กำลังทำความสะอาดรูปภาพ...",
        "result_header": "📄 ข้อความที่อ่านได้:",
        "download_btn": "📥 ดาวน์โหลดข้อความ",
        "error_msg": "AI อ่านข้อความไม่ออก ลองปรับค่าประมวลผลใหม่",
        "info_msg": "💡 กรุณาอัปโหลดรูปภาพเพื่อเริ่มการทดสอบ",
    },
    "EN": {
        "title": "📸 AI Multi-Language Text Scanner",
        "subtitle": "Pro Version: Noise Reduction & Text Sharpening for High Accuracy",
        "settings": "⚙️ Processing Settings",
        "select_lang_ocr": "Select OCR Language:",
        "noise_label": "Noise Removal Level",
        "upscale_label": "Upscale Factor",
        "upload_label": "Upload Image...",
        "src_img": "🖼️ Original Image",
        "proc_img": "✨ AI Processed Image",
        "scanning": "Extracting text...",
        "cleaning": "Cleaning image...",
        "result_header": "📄 Scanned Result:",
        "download_btn": "📥 Download Result",
        "error_msg": "No text detected. Try adjusting settings.",
        "info_msg": "💡 Please upload an image to start testing.",
    }
}

# --- Sidebar: UI Language Switcher ---
st.sidebar.subheader("🌐 UI Language / เลือกภาษาหน้าเว็บ")
ui_lang = st.sidebar.radio("Select Interface Language:", ("TH", "EN"), horizontal=True)
txt = ui_strings[ui_lang]

# Custom CSS
st.markdown(f"""
    <style>
    .main {{ background-color: #f5f5f5; }}
    .stTextArea textarea {{ font-size: 18px !important; color: #1e1e1e; }}
    </style>
    """, unsafe_allow_html=True)

st.title(txt["title"])
st.write(txt["subtitle"])

# --- Sidebar: OCR Settings ---
st.sidebar.divider()
st.sidebar.header(txt["settings"])

ocr_lang_options = {
    "ไทย + English": "tha+eng",
    "English Only": "eng",
    "ภาษาจีน (ตัวย่อ)": "chi_sim",
    "ภาษาจีน (ตัวเต็ม)": "chi_tra"
}
selected_option = st.sidebar.selectbox(txt["select_lang_ocr"], list(ocr_lang_options.keys()))
selected_lang = ocr_lang_options[selected_option]

clean_level = st.sidebar.slider(txt["noise_label"], 1, 5, 3)
upscale_factor = st.sidebar.slider(txt["upscale_label"], 1.0, 3.0, 1.5, 0.5)

uploaded_file = st.sidebar.file_uploader(txt["upload_label"], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(txt["src_img"])
        st.image(image, use_container_width=True)

    # --- Processing Logic ---
    with st.spinner(txt["cleaning"]):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if upscale_factor > 1.0:
            gray = cv2.resize(gray, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 10
        )

        if clean_level > 1:
            k_size = (clean_level // 2) * 2 + 1
            binary = cv2.medianBlur(binary, k_size)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    with col2:
        st.subheader(txt["proc_img"])
        st.image(processed_img, use_container_width=True, channels="GRAY")

    # --- OCR Process ---
    st.divider()
    with st.spinner(txt["scanning"]):
        try:
            custom_config = r'--oem 3 --psm 3'
            text = pytesseract.image_to_string(processed_img, lang=selected_lang, config=custom_config)
            
            if text.strip():
                st.subheader(txt["result_header"])
                st.text_area("Result", text, height=450, label_visibility="hidden")
                st.download_button(txt["download_btn"], text.encode('utf-8'), "result.txt", "text/plain")
            else:
                st.error(txt["error_msg"])
        
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info(txt["info_msg"])
