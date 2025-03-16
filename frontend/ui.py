import streamlit as st
from backend.model import load_model, preprocess_image, get_prediction
from PIL import Image
import numpy as np
import os
from frontend.utils import find_file, load_html_template, get_html_section, set_background


def show_frontend():
    """Hiển thị giao diện người dùng của Streamlit"""
    st.set_page_config(
        page_title="Potato Leaf Disease Classifier",
        page_icon="🥔",
        layout="wide",
    )
    
    # Load mẫu HTML
    html_template = load_html_template()
    
    # Apply custom CSS
    css_path = 'frontend/style.css'
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Apply background CSS
    bg_css_path = 'frontend/background.css'
    if os.path.exists(bg_css_path):
        with open(bg_css_path, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Apply CSS for tabs and metrics từ template
    st.markdown(get_html_section(html_template, "tab-styling"), unsafe_allow_html=True)
    st.markdown(get_html_section(html_template, "metrics-styling"), unsafe_allow_html=True)
    
    # Container wrapper cho header
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    
    # Hiển thị logo và tiêu đề sử dụng cột của Streamlit (trên cùng một hàng)
    col1, col2 = st.columns([1, 5])
    
    with col1:
        # Hiển thị logo với frame đẹp
        logo_path = 'frontend/Logo/Logo.png'
        try:
            if os.path.exists(logo_path):
                st.markdown('<div class="logo-frame">', unsafe_allow_html=True)
                image = Image.open(logo_path)
                st.image(image, width=120)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="logo-frame"><span style="font-size: 60px; margin: 10px;">🍃</span></div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown('<div class="logo-frame"><span style="font-size: 60px; margin: 10px;">🍃</span></div>', unsafe_allow_html=True)
    
    with col2:
        # Hiển thị tiêu đề với phong cách nâng cao
        st.markdown('<div class="title-col">', unsafe_allow_html=True)
        st.markdown("<h1 class='app-title'>Potato Leaf Disease Classifier🔎 </h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>AI-powered diagnosis for potato plant health 🍃</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Thêm đường gạch ngang gradient
    st.markdown("<div class='header-decoration'></div>", unsafe_allow_html=True)
    
    # Đóng container wrapper
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main app container
    with st.container():
        # Sử dụng template HTML cho welcome box
        st.markdown(get_html_section(html_template, "welcome-box"), unsafe_allow_html=True)

        # Initialize model in session state
        if 'model' not in st.session_state:
            try:
                with st.spinner('Loading model... Please wait a moment'):
                    # Đơn giản hóa đường dẫn model dựa trên cấu trúc thư mục thực tế
                    model_path = 'models/efficientnetB0.h5'
                    
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Model file not found at {model_path}")
                        
                    st.session_state.model = load_model(model_path)
                    st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"⚠️ Error initializing model: {str(e)}")
                st.stop()
        
        # Create tabs
        tab1, tab2 = st.tabs(["🔍 Leaf Analysis", "📊 How It Works"])
        
        with tab1:
            # File uploader with instructions from template
            st.markdown(get_html_section(html_template, "upload-instructions"), unsafe_allow_html=True)
            
            # Định nghĩa ảnh mẫu
            sample_diseases = ["Bacteria", "Nematode", "Pest"]
            sample_image_paths = {
                "Bacteria": "frontend/Sample Images/Bacteria.jpg",
                "Nematode": "frontend/Sample Images/nematode.jpg",
                "Pest": "frontend/Sample Images/Pest.jpg"
            }
            
            # Tạo hai cách phân tích: tải lên hoặc chọn mẫu
            analysis_mode = st.radio(
                "Choose analysis method:",
                ["Upload your own image", "Use sample images"],
                horizontal=True
            )
            
            if analysis_mode == "Upload your own image":
                # Hiển thị file uploader
                uploaded_file = st.file_uploader("Choose a leaf image...", type=['jpg', 'jpeg', 'png'])
                
                # Nếu có file được tải lên, phân tích nó
                if uploaded_file:
                    image_to_analyze = uploaded_file
                    analyze_image = True
                else:
                    analyze_image = False
            else:
                # Hiển thị ảnh mẫu
                st.write("**Select a sample image to analyze:**")
                
                sample_cols = st.columns(3)
                selected_sample = None
                
                for i, col in enumerate(sample_cols):
                    with col:
                        disease = sample_diseases[i]
                        image_path = sample_image_paths[disease]
                        
                        if os.path.exists(image_path):
                            # Hiển thị ảnh mẫu
                            col.markdown(f'<div class="sample-image-container">', unsafe_allow_html=True)
                            col.image(image_path, use_container_width=True)
                            col.markdown(f'</div>', unsafe_allow_html=True)
                            
                            # Thêm nút Analyze cho từng ảnh
                            if col.button("Analyze", key=f"analyze_{disease}"):
                                selected_sample = image_path
                
                # Nếu ảnh mẫu được chọn, phân tích nó
                if selected_sample:
                    image_to_analyze = selected_sample
                    analyze_image = True
                else:
                    analyze_image = False
            
            # Phân tích ảnh nếu có ảnh để phân tích
            if analyze_image:
                # Tạo columns cho layout
                col1, col2 = st.columns([2, 3])
                
                # Column 1: Hiển thị ảnh
                with col1:
                    st.markdown(get_html_section(html_template, "image-header"), unsafe_allow_html=True)
                    
                    # Hiển thị ảnh sẽ phân tích
                    if isinstance(image_to_analyze, str):  # Nếu là đường dẫn file
                        image = Image.open(image_to_analyze)
                    else:  # Nếu là uploaded file
                        image = Image.open(image_to_analyze)
                    
                    st.image(image, use_container_width=True)
                
                # Column 2: Hiển thị kết quả
                with col2:
                    st.markdown(get_html_section(html_template, "analysis-header"), unsafe_allow_html=True)
                    
                    with st.spinner("Analyzing leaf... Please wait"):
                        try:
                            # Xử lý ảnh và dự đoán
                            if isinstance(image_to_analyze, str):  # Nếu là đường dẫn file
                                img_array = preprocess_image(image_to_analyze)
                            else:  # Nếu là uploaded file
                                img_array = preprocess_image(image_to_analyze)
                                
                            predicted_label, confidence, top_3_predictions = get_prediction(st.session_state.model, img_array)
                            
                            # Hiển thị kết quả
                            result_card = get_html_section(html_template, "result-card-template").replace("{predicted_label}", predicted_label)
                            st.markdown(result_card, unsafe_allow_html=True)
                            
                            # Chỉ số độ tin cậy - chỉ hiển thị số liệu, không hiển thị thanh tiến trình
                            st.write(f"Confidence level: {confidence*100:.1f}%")
                            
                            # Top 3 dự đoán
                            st.markdown(get_html_section(html_template, "top3-header"), unsafe_allow_html=True)
                            
                            for disease, prob in top_3_predictions:
                                # Hiển thị tên bệnh và phần trăm trực tiếp bằng HTML tùy chỉnh
                                st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #f0f0f0;">
                                    <span style="font-weight: 500; color: #388E3C; font-size: 18px;">{disease}</span>
                                    <span style="font-weight: 500; font-size: 18px;">{prob*100:.1f}%</span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Thông tin bệnh
                            with st.expander("📋 Disease Information & Treatment"):
                                disease_info = {
                                    "Bacteria": {
                                        "desc": "Bacterial diseases in potato plants often cause wilting, rotting, and discoloration. They can spread quickly in warm, humid conditions.",
                                        "treatment": "Apply copper-based bactericides and ensure good air circulation between plants. Remove and destroy infected plants to prevent spread. Practice crop rotation to reduce soil-borne bacteria."
                                    },
                                    "Fungi": {
                                        "desc": "Fungal infections can cause spots, lesions, and tissue death.",
                                        "treatment": "Regular fungicide application and avoiding overhead watering."
                                    },
                                    "Healthy": {
                                        "desc": "The leaf appears healthy with no visible signs of disease.",
                                        "treatment": "Continue with regular care and monitoring."
                                    },
                                    "Nematode": {
                                        "desc": "Nematode infections affect the roots and can cause stunted growth and yellowing of leaves.",
                                        "treatment": "Crop rotation and nematicide application when necessary."
                                    },
                                    "Pest": {
                                        "desc": "Pest damage is characterized by chewing marks, holes, and irregular patterns on leaves.",
                                        "treatment": "Apply appropriate insecticides or introduce beneficial insects."
                                    },
                                    "Phytopthora": {
                                        "desc": "A serious water mold infection that can cause late blight, leading to rapid plant death if untreated.",
                                        "treatment": "Fungicides containing metalaxyl or chlorothalonil, and ensuring proper drainage."
                                    },
                                    "Virus": {
                                        "desc": "Viral infections often cause mottling, stunting, and leaf deformation.",
                                        "treatment": "No cure available. Remove and destroy infected plants to prevent spread."
                                    }
                                }
                                
                                # Ensure the disease exists in our dictionary
                                if predicted_label in disease_info:
                                    # Show disease description and treatment
                                    st.markdown(f"<h4 style='color: #388E3C;'>🎯 About {predicted_label}:</h4>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='background-color: rgba(76, 175, 80, 0.05); padding: 15px; border-radius: 8px;'>{disease_info[predicted_label]['desc']}</p>", unsafe_allow_html=True)
                                    
                                    st.markdown(f"<h4 style='color: #689F38; margin-top: 20px;'>💊 Recommended Treatment:</h4>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='background-color: rgba(139, 195, 74, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #689F38;'>{disease_info[predicted_label]['treatment']}</p>", unsafe_allow_html=True)
                                else:
                                    st.warning(f"No detailed information available for {predicted_label}")
                                    st.write("General recommendation: Consult with a plant pathologist for specific treatment options.")
                        
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")
                            st.write("Please try again with a different image")
                
        with tab2:
            # Sử dụng các template cho tab "How It Works"
            st.markdown(get_html_section(html_template, "how-model-works"), unsafe_allow_html=True)
            st.markdown(get_html_section(html_template, "model-explanation"), unsafe_allow_html=True)
            st.markdown(get_html_section(html_template, "process-explanation"), unsafe_allow_html=True)
            
            # Add a metrics section
            st.markdown(get_html_section(html_template, "model-performance"), unsafe_allow_html=True)
            
            metric1, metric2, metric3, metric4 = st.columns(4)
            with metric1:
                st.metric(label="Accuracy", value="94.5%")
            with metric2:
                st.metric(label="Precision", value="93.2%")
            with metric3:
                st.metric(label="Recall", value="91.8%")
            with metric4:
                st.metric(label="F1", value="90%")

    # Footer with template
    st.markdown(get_html_section(html_template, "footer-text"), unsafe_allow_html=True)