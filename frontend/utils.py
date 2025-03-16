import streamlit as st
import base64
import os
from PIL import Image

def get_base64_of_bin_file(bin_file):
    """Chuyển đổi file nhị phân sang chuỗi base64"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    """Đặt hình nền cho ứng dụng Streamlit"""
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def find_file(possible_paths, debug=False):
    """Tìm kiếm file từ danh sách đường dẫn có thể"""
    for path in possible_paths:
        if os.path.exists(path):
            if debug:
                print(f"Found file at: {path}")
            return path
    if debug:
        print(f"Could not find file in paths: {possible_paths}")
    return None

def encode_image_to_base64(image_path):
    """Mã hóa file ảnh thành chuỗi base64"""
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
            return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None

def load_html_template(template_file="frontend/templates.html"):
    """Đọc và trả về toàn bộ nội dung của file HTML template"""
    try:
        with open(template_file, "r", encoding="utf-8") as file:
            content = file.read()
            return content
    except Exception as e:
        print(f"Không thể đọc file template HTML: {str(e)}")
        return None

def get_html_section(html_content, section_id):
    """Trích xuất phần HTML dựa trên ID của phần tử"""
    if not html_content:
        return ""
    
    start_tag = f'<div id="{section_id}"'
    end_tag = '</div>'
    
    # Nếu là thẻ h3, h4...
    if section_id.startswith("h"):
        tag_type = section_id.split("-")[0]  # Lấy loại thẻ (h3, h4,...)
        start_tag = f'<{tag_type} id="{section_id}"'
        end_tag = f'</{tag_type}>'
    
    # Nếu là phong cách CSS
    if section_id.endswith("styling"):
        start_tag = f'<style id="{section_id}"'
        end_tag = '</style>'
    
    try:
        start_index = html_content.find(start_tag)
        if start_index == -1:
            return ""
        
        # Tìm vị trí kết thúc của phần tử
        end_index = html_content.find(end_tag, start_index)
        if end_index == -1:
            return ""
        
        # Trả về phần HTML hoàn chỉnh bao gồm cả thẻ đóng
        return html_content[start_index:end_index + len(end_tag)]
    except Exception as e:
        print(f"Lỗi khi trích xuất phần HTML: {str(e)}")
        return "" 