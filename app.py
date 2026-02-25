import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="AI Multi-Lang OCR (Improved)",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stTextArea textarea { font-size: 18px !important; font-family: 'Tahoma', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("📸 AI Multi-Language Text Scanner")
st.write("เวอร์ชันปรับปรุง: สู้เงาและเพิ่มความคมชัดสำหรับตัวหนังสือขนาดเล็ก")

# --- ส่วน Sidebar ---
st.sidebar.header("⚙️ การตั้งค่า")

lang_options = {
    "ไทย + English": "tha+eng",
    "English Only": "eng",
    "ภาษาจีน (ตัวย่อ)": "chi_sim",
    "ภาษาจีน (ตัวเต็ม)": "chi_tra",
    "ไทย + จีน": "tha+chi_sim"
}
selected_option = st.sidebar.selectbox("เลือกภาษาที่แสดงในภาพ:", list(lang_options.keys()))
selected_lang = lang_options[selected_option]

# เพิ่มความสามารถในการขยายภาพ (จำเป็นมากสำหรับรูปที่คุณ Nick ส่งมา)
upscale_factor = st.sidebar.slider("ขยายขนาดภาพ (Upscale)", 1.0, 3.0, 2.0, 0.5)

uploaded_file = st.sidebar.file_uploader("อัปโหลดรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ ภาพต้นฉบับ")
        st.image(image, use_container_width=True)

    # --- Improved Image Enhancement Logic ---
    with st.spinner('กำลังประมวลผลภาพด้วย Logic ใหม่...'):
        # 1. แปลงเป็นขาวดำ
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 2. ขยายขนาดภาพ (Upscaling) - ช่วยให้ตัวหนังสือเล็กๆ ชัดขึ้น
        if upscale_factor > 1.0:
            width = int(gray.shape[1] * upscale_factor)
            height = int(gray.shape[0] * upscale_factor)
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # 3. ใช้ Adaptive Threshold แทน Otsu เพื่อสู้กับ "เงา" ในรูปภาพ
        # วิธีนี้จะคำนวณแสงแยกเป็นโซนๆ ทำให้ตัวหนังสือไม่หายไปในพื้นที่มืด
        processed_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 10
        )
        
        # 4. ลดจุดรบกวน (Noise Reduction)
        kernel = np.ones((1, 1), np.uint8)
        processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel)

    with col2:
        st.subheader("✨ ภาพหลังปรับแต่ง (สู้เงา + ขยาย)")
        st.image(processed_img, use_container_width=True, channels="GRAY")

    # --- Improved OCR Process ---
    st.divider()
    with st.spinner('AI กำลังวิเคราะห์ข้อความ...'):
        try:
            # ใช้ PSM 3 เพื่อให้ AI วิเคราะห์โครงสร้างหน้ากระดาษอัตโนมัติ
            custom_config = r'--oem 3 --psm 3'
            text = pytesseract.image_to_string(processed_img, lang=selected_lang, config=custom_config)
            
            if text.strip():
                st.subheader("📄 ข้อความที่ตรวจพบ:")
                st.text_area("Result", text, height=450, label_visibility="hidden")
                st.download_button("📥 ดาวน์โหลดข้อความ", text.encode('utf-8'), "result.txt", "text/plain")
            else:
                st.warning("⚠️ AI ยังไม่พบข้อความ ลองปรับค่า 'ขยายขนาดภาพ (Upscale)' ในแถบด้านข้างให้สูงขึ้น")
        
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")

else:
    st.info("💡 อัปโหลดรูปภาพที่แถบด้านข้างเพื่อเริ่มสแกน")
