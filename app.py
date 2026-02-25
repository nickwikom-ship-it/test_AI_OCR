import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

# --- Web Configuration ---
st.set_page_config(page_title="AI Multi-Lang OCR Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { font-size: 18px !important; color: #1e1e1e; line-height: 1.6; }
    .stSelectbox label { font-weight: bold; color: #007bff; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 AI Multi-Language Text Scanner")
st.write("Advanced Version: Noise Reduction & Deskewing for Better Accuracy")

# --- Main UI Controls ---
col_lang, col_file = st.columns([1, 2])

with col_lang:
    # Language Selection on Main Page
    lang_options = {
        "🇹🇭 Thai + English": "tha+eng",
        "🇺🇸 English Only": "eng",
        "🇨🇳 Chinese (Simplified)": "chi_sim",
        "🇭🇰 Chinese (Traditional)": "chi_tra"
    }
    selected_option = st.selectbox("🌐 Select Target Language:", list(lang_options.keys()))
    selected_lang = lang_options[selected_option]

with col_file:
    uploaded_file = st.file_uploader("📂 Upload an image to scan...", type=["jpg", "jpeg", "png"])

# --- Sidebar for Advanced Settings ---
st.sidebar.header("⚙️ Image Pre-processing")
upscale_factor = st.sidebar.slider("Upscale Factor", 1.0, 3.0, 1.5, 0.5)
clean_level = st.sidebar.slider("Noise Removal Level", 1, 7, 3, 2)
auto_rotate = st.sidebar.checkbox("Auto Deskew (Rotate Straight)", value=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))
    
    col_src, col_proc = st.columns(2)

    with col_src:
        st.subheader("🖼️ Original Image")
        st.image(image, use_container_width=True)

    # --- Advanced Cleaning Logic ---
    with st.spinner('AI is cleaning the image...'):
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 2. Upscaling
        if upscale_factor > 1.0:
            gray = cv2.resize(gray, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
        
        # 3. Auto-Deskew
        if auto_rotate:
            coords = np.column_stack(np.where(gray < 127))
            if coords.size > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45: angle = -(90 + angle)
                else: angle = -angle
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # 4. Bilateral Filter (Smooth background while keeping edges sharp)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # 5. Adaptive Thresholding (Handle shadows)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 10
        )

        # 6. Median Blur (Remove small dots/noise)
        if clean_level > 1:
            k_size = (clean_level // 2) * 2 + 1
            processed_img = cv2.medianBlur(binary, k_size)
        else:
            processed_img = binary

    with col_proc:
        st.subheader("✨ Enhanced Image (Ready for OCR)")
        st.image(processed_img, use_container_width=True, channels="GRAY")

    # --- OCR Process ---
    st.divider()
    with st.spinner(f'Extracting {selected_option} text...'):
        try:
            # Config: OEM 3 (LSTM) | PSM 3 (Auto segmentation)
            custom_config = r'--oem 3 --psm 3'
            text = pytesseract.image_to_string(processed_img, lang=selected_lang, config=custom_config)
            
            if text.strip():
                st.subheader("📄 Scanned Result:")
                st.text_area("OCR Output", text, height=450, label_visibility="collapsed")
                
                # Download Button
                st.download_button(
                    label="📥 Download Result (.txt)",
                    data=text.encode('utf-8'),
                    file_name="ocr_result.txt",
                    mime="text/plain"
                )
            else:
                st.error("⚠️ No text detected. Try adjusting 'Upscale' or 'Noise Removal' in the sidebar.")
        
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("💡 Get started by uploading an image from the top.")
    with st.expander("Pro Tips for better results"):
        st.write("""
        1. **Language Match:** Ensure the selected language matches the text in your image.
        2. **Upscale:** Use a higher upscale factor (2.0x+) if the text is very small.
        3. **Noise Removal:** If the background is grainy, increase the noise removal level.
        """)
