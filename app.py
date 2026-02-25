import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Multi-Lang OCR Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { font-size: 18px !important; color: #1e1e1e; line-height: 1.6; }
    .stSelectbox label { font-weight: bold; color: #007bff; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 AI Multi-Language Text Scanner")
st.write("เวอร์ชันอัปเกรด: เพิ่มระบบหมุนภาพตรงอัตโนมัติ และปุ่มสลับภาษา")

# --- ส่วนควบคุมหลักบนหน้าเว็บ (Main UI) ---
col_lang, col_file = st.columns([1, 2])

with col_lang:
    # ปุ่มเปลี่ยนภาษาหลัก
    lang_options = {
        "🇹🇭 ไทย + English": "tha+eng",
        "🇺🇸 English Only": "eng",
        "🇨🇳 จีน (ตัวย่อ)": "chi_sim",
        "🇭🇰 จีน (ตัวเต็ม)": "chi_tra"
    }
    selected_option = st.selectbox("🌐 เลือกภาษาที่สแกน:", list(lang_options.keys()))
    selected_lang = lang_options[selected_option]

with col_file:
    uploaded_file = st.file_uploader("📂 อัปโหลดรูปภาพที่ต้องการสแกน...", type=["jpg", "jpeg", "png"])

# --- Sidebar สำหรับปรับแต่งค่าเชิงลึก ---
st.sidebar.header("🛠️ เครื่องมือปรับแต่งภาพ")
upscale = st.sidebar.slider("ขยายขนาดภาพ (Upscale)", 1.0, 3.0, 1.5, 0.5)
noise_level = st.sidebar.slider("ลบจุดรบกวน (Noise Removal)", 1, 7, 3, 2)
auto_rotate = st.sidebar.checkbox("หมุนภาพให้ตรงอัตโนมัติ (Deskew)", value=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))
    
    col_src, col_proc = st.columns(2)

    with col_src:
        st.subheader("🖼️ รูปต้นฉบับ")
        st.image(image, use_container_width=True)

    # --- ขั้นตอนการประมวลผล (Advanced Image Processing) ---
    with st.spinner('กำลังปรับแต่งภาพให้ชัดเจน...'):
        # 1. แปลงเป็นขาวดำ
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2. ขยายขนาดภาพ
        if upscale > 1.0:
            gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

        # 3. หมุนภาพให้ตรง (Deskewing)
        if auto_rotate:
            coords = np.column_stack(np.where(gray < 127))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45: angle = -(90 + angle)
            else: angle = -angle
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # 4. ลดจุดรบกวนด้วย Bilateral Filter (รักษาขอบตัวอักษร)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # 5. ทำ Adaptive Threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 10
        )

        # 6. Median Blur ลบจุดดำเล็กๆ (Noise)
        processed_img = cv2.medianBlur(binary, noise_level)

    with col_proc:
        st.subheader("✨ ภาพหลังปรับแต่ง (พร้อมสแกน)")
        st.image(processed_img, use_container_width=True, channels="GRAY")

    # --- ขั้นตอน OCR ---
    st.divider()
    with st.spinner(f'กำลังอ่านข้อความภาษา {selected_option}...'):
        try:
            # --oem 3: Default (LSTM) | --psm 3: Auto page segmentation
            config = r'--oem 3 --psm 3'
            text = pytesseract.image_to_string(processed_img, lang=selected_lang, config=config)
            
            if text.strip():
                st.subheader(f"📄 ผลลัพธ์ข้อความ ({selected_option}):")
                st.text_area("", text, height=450, label_visibility="collapsed")
                
                # ปุ่มดาวน์โหลดผลลัพธ์
                st.download_button(
                    label="📥 ดาวน์โหลดไฟล์ข้อความ (.txt)",
                    data=text.encode('utf-8'),
                    file_name="ocr_result.txt",
                    mime="text/plain"
                )
            else:
                st.error("❌ AI ไม่พบข้อความที่ชัดเจน ลองปรับ 'Upscale' หรือ 'Noise Removal' ในแถบด้านข้าง")
        
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
            st.info("ตรวจสอบว่าได้ติดตั้งภาษาใน packages.txt ครบถ้วนแล้ว")

else:
    st.info("💡 เริ่มต้นใช้งานโดยการอัปโหลดรูปภาพที่ด้านบน")
