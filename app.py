import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Multi-Lang OCR (Pro)", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stTextArea textarea { font-size: 18px !important; color: #1e1e1e; }
    </style>
    """, unsafe_allow_html=True)

st.title("📸 AI Multi-Language Text Scanner (Pro-Cleaning)")
st.write("เวอร์ชันแก้ไข: ลดสัญญาณรบกวนและเชื่อมเส้นตัวอักษรเพื่อให้อ่านภาษาไทย/จีนได้แม่นยำขึ้น")

# --- Sidebar ---
st.sidebar.header("⚙️ การตั้งค่าประมวลผล")

lang_options = {
    "ไทย + English": "tha+eng",
    "English Only": "eng",
    "ภาษาจีน (ตัวย่อ)": "chi_sim",
    "ภาษาจีน (ตัวเต็ม)": "chi_tra"
}
selected_option = st.sidebar.selectbox("เลือกภาษา:", list(lang_options.keys()))
selected_lang = lang_options[selected_option]

# Slider สำหรับปรับจูนกรณีรูปยังอ่านไม่ออก
clean_level = st.sidebar.slider("ระดับการลบจุดรบกวน (Noise Removal)", 1, 5, 3)
upscale_factor = st.sidebar.slider("ขยายขนาดภาพ", 1.0, 3.0, 1.5, 0.5)

uploaded_file = st.sidebar.file_uploader("อัปโหลดรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ ภาพต้นฉบับ")
        st.image(image, use_container_width=True)

    # --- New Advanced Cleaning Logic ---
    with st.spinner('AI กำลังทำความสะอาดรูปภาพ...'):
        # 1. แปลงเป็นขาวดำ
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 2. ขยายขนาดภาพ
        if upscale_factor > 1.0:
            gray = cv2.resize(gray, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
        
        # 3. ลด Noise ครั้งที่ 1 (Bilateral Filter) - ช่วยให้ขอบตัวอักษรชัดแต่ผิวพื้นหลังเรียบ
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # 4. Adaptive Thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 10
        )

        # 5. ลบจุดรบกวน (Median Blur) - สำคัญมากในการแก้ปัญหาตัวอักษรต่างดาว
        if clean_level > 1:
            k_size = (clean_level // 2) * 2 + 1 # ต้องเป็นเลขคี่
            binary = cv2.medianBlur(binary, k_size)

        # 6. เชื่อมเส้นตัวอักษร (Dilation/Erosion) - ทำให้หัวตัวนสือชัดขึ้น
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    with col2:
        st.subheader("✨ ภาพหลังการ Clean (เหมาะสำหรับ OCR)")
        st.image(processed_img, use_container_width=True, channels="GRAY")

    # --- OCR Process ---
    st.divider()
    with st.spinner('กำลังแปลงภาพเป็นข้อความ...'):
        try:
            # ใช้ config ที่เน้นการอ่านภาษาไทย
            # --psm 3 (Auto page segmentation)
            custom_config = r'--oem 3 --psm 3'
            text = pytesseract.image_to_string(processed_img, lang=selected_lang, config=custom_config)
            
            if text.strip():
                st.subheader("📄 ข้อความที่อ่านได้:")
                st.text_area("Result", text, height=450, label_visibility="hidden")
                st.download_button("📥 ดาวน์โหลดข้อความ", text.encode('utf-8'), "result.txt", "text/plain")
            else:
                st.error("AI อ่านข้อความไม่ออก ลองปรับ 'ระดับการลบจุดรบกวน' ให้ลดลง หรือเพิ่ม 'ขยายขนาดภาพ'")
        
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("💡 กรุณาอัปโหลดรูปภาพเพื่อเริ่มการทดสอบ")
