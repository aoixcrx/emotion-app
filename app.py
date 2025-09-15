import streamlit as st
import torch
from collections import OrderedDict
from torch.serialization import safe_globals
import lightning.fabric.wrappers
from pytorch_lightning import LightningModule
import plotly.express as px
from PIL import Image
from prediction import pred_class
import numpy as np
import plotly.graph_objects as go
import os
import base64
import io
import gdown
import timm
from torchvision import transforms

# Fix for PyTorch 2.2.0 compatibility
try:
    from torch.cuda.amp import GradScaler
except ImportError:
    try:
        from torch.amp import GradScaler
    except ImportError:
        # Fallback for older versions
        GradScaler = None

# Page Configuration
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to encode image to base64
@st.cache_data
def get_base64_of_bin_file(bin_file):
    """Convert binary file to base64 string"""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def create_css_with_banner():
    banner_paths = ["banner01.png", "images/banner01.png", "assets/banner01.png", "./banner01.png"]
    banner_base64 = None
    banner_found = False

    for path in banner_paths:
        if os.path.exists(path):
            banner_base64 = get_base64_of_bin_file(path)
            if banner_base64:
                banner_found = True
                break

    # CSS สำหรับ background
    if banner_base64:
        banner_bg = f"""
        background: url(data:image/png;base64,{banner_base64}) no-repeat center center;
        background-size: cover;
        background-position: center center;
        """
    else:
        # ใช้ raw GitHub image แทน gradient
        banner_url = "https://raw.githubusercontent.com/aoixcrx/emotion-app/main/banner01.png"
        banner_bg = f"""
        background: url("{banner_url}") no-repeat center center;
        background-size: cover;
        background-position: center center;
        """

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;900&family=Inter:wght@300;400;500;600&display=swap');

/* Reset */
* {{ 
    box-sizing: border-box; 
    margin: 0; 
    padding: 0; 
    font-family: 'Poppins', sans-serif; 
}}

/* Keyframes for animations */
@keyframes fadeInUp {{
    from {{
        opacity: 0;
        transform: translateY(30px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes slideInRight {{
    from {{
        opacity: 0;
        transform: translateX(50px);
    }}
    to {{
        opacity: 1;
        transform: translateX(0);
    }}
}}

@keyframes bounce {{
    0%, 20%, 50%, 80%, 100% {{
        transform: translateY(0);
    }}
    40% {{
        transform: translateY(-10px);
    }}
    60% {{
        transform: translateY(-5px);
    }}
}}

@keyframes glow {{
    0% {{
        box-shadow: 0 0 5px rgba(255, 255, 255, 0.2);
    }}
    50% {{
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.4), 0 0 30px rgba(255, 255, 255, 0.2);
    }}
    100% {{
        box-shadow: 0 0 5px rgba(255, 255, 255, 0.2);
    }}
}}

@keyframes float {{
    0% {{ transform: rotate(0deg) translateY(0px); }}
    50% {{ transform: rotate(180deg) translateY(-20px); }}
    100% {{ transform: rotate(360deg) translateY(0px); }}
}}

/* Hide Streamlit default elements */
.stApp > header {{
    display: none;
}}

.stApp {{
    margin-top: 0 !important;
    background-color: #000000 !important;
}}

/* ซ่อน sidebar */
.stSidebar {{
    display: none !important;
}}

/* Main container black background */
.main {{
    background-color: #000000 !important;
}}

/* Logo Bar - Dark */
.logo-bar {{
    background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #112e63 100%);
    padding: 12px 0;
    text-align: center;
    width: 100%;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.8);
    position: relative;
    z-index: 10;
    margin: 0;
    border-bottom: 1px solid #333;
}}

.logo-content {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 15px;
    color: #ffffff;
    font-size: 1 rem;
    font-weight: 700;
    opacity: 0.85;
}}

.brain-icon {{
    font-size: 1.8rem;
    color: #fff !important;
}}

/* Banner Section - Dark */
.banner-section {{
    width: 100%;
    height: 450px;
    {banner_bg}
    position: relative;
    margin: 0;
    overflow: hidden;
    display: flex;
    align-items: center;
}}

.banner-content {{
    position: relative;
    z-index: 2;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
    width: 100%;
}}

.banner-text {{
    max-width: 60%;
    text-align: left;
    line-height: 1.2;
    animation: fadeInUp 1.2s ease-out;
}}

.banner-text h1 {{
    font-size: 3.5 rem; 
    font-weight: 900;
    color: #ffffff;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
    letter-spacing: -1px;
    white-space: nowrap;
    line-height: 1.1;
}}

.banner-text h2 {{
    font-size: 2rem; 
    font-weight: 700;
    color: #ffffff;
    margin: 2px 0;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
    line-height: 1.2;
}}

.banner-text p {{
    font-size: 1.2rem;
    line-height: 1.4;
    color: #cccccc !important;
    margin: 8px 0 0 0;
    max-width: 500px;
}}

/* Content Container - Dark */
.content-container {{
    max-width: 100%;
    margin: 0;
    padding: 0rem 0;
    background: #000000;
}}

/* Enhanced About Section - Without Cards */
.about-section {{
    background: transparent;
    padding: 4rem 2rem;
    margin: 2rem 0;
    position: relative;
}}

.about-container {{
    max-width: 1200px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4rem;
    align-items: start;
}}

.about-main {{
    position: relative;
    grid-column: 1 / -1;
    justify-self: center;
    text-align: center;
}}

.about-title {{
     font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff !important;  /* ขาวสด */
    margin-bottom: 1.2rem;
    position: relative;
    text-align: center;
    letter-spacing: 1px;

    /* ยกเลิกทุก effect เดิมที่ทำให้มันไม่ขาว */
    background: none !important;
    -webkit-background-clip: unset !important;
    -webkit-text-fill-color: #ffffff !important; 
    background-clip: unset !important;
}}

.about-title::after {{
    content: '';
    display: block;
    margin: 0.7rem auto 0 auto;
    width: 160px;
    height: 3px;
    background: linear-gradient(90deg, #7db3d3 0%, #a8d0e6 100%);
    border-radius: 2px;
    box-shadow: 0 0 12px #7db3d3, 0 0 2px #a8d0e6;
    opacity: 0.85;
}}

.about-description {{
    font-size: 1.2rem;
    line-height: 1.8;
    color: #cccccc;
    margin-bottom: 2rem;
    animation: fadeInUp 1s ease-out 0.3s both;
}}

.emotions-sidebar {{
    position: relative;
}}

.emotions-container {{
        background: none !important;
    border: none !important;
    box-shadow: none !important;
    padding: 2rem 0 1rem 0 !important;
}}

.emotions-container:hover {{
    transform: translateY(-8px);
    box-shadow: 0 20px 50px rgba(255, 255, 255, 0.1);
    border-color: #666;
}}

.emotions-container::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #666, #999, #ccc, #999, #666);
    background-size: 200% 100%;
    animation: slideInRight 3s ease-in-out infinite;
}}

.emotions-title {{
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff !important;  /* ขาวสด */
    margin-bottom: 1.2rem;
    position: relative;
    text-align: center;
    letter-spacing: 1px;

    /* ยกเลิกทุก effect เดิมที่ทำให้มันไม่ขาว */
    background: none !important;
    -webkit-background-clip: unset !important;
    -webkit-text-fill-color: #ffffff !important; 
    background-clip: unset !important;
}}

.emotions-title::after {{
    content: '';
    display: block;
    margin: 0.7rem auto 0 auto;
    width: 160px;
    height: 3px;
    background: linear-gradient(90deg, #7db3d3 0%, #a8d0e6 100%);
    border-radius: 2px;
    box-shadow: 0 0 12px #7db3d3, 0 0 2px #a8d0e6;
    opacity: 0.85;
}}

.emotions-grid {{
       display: flex;
    justify-content: center;
    gap: 1.2rem;           /* ลดช่องว่างระหว่างอิโมจิ */
    margin-top: 1rem;
    flex-wrap: wrap;
    max-width: 400px;      /* จำกัดความกว้างสูงสุด */
    margin-left: auto;
    margin-right: auto;
}}

.emotion-item {{
        min-width: 60px;       /* จำกัดความกว้างขั้นต่ำแต่ไม่กว้างเกิน */
    padding: 0 0.2rem; 
}}

.emotion-item:hover {{
    transform: translateY(-10px) scale(1.05);
    border-color: #666;
    box-shadow: 0 15px 30px rgba(255, 255, 255, 0.1);
}}

.emotion-item::before {{
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: radial-gradient(circle, rgba(255,255,255,0.1), transparent);
    transition: all 0.6s ease;
    transform: translate(-50%, -50%);
    border-radius: 50%;
}}

.emotion-item:hover::before {{
    width: 200px;
    height: 200px;
}}

.emotion-emoji {{
    font-size: 2.8rem;
    display: block;
    margin-bottom: 0.5rem;
    animation: emoji-bounce 1.6s infinite alternate cubic-bezier(.5,1.5,.5,1);
}}
@keyframes emoji-bounce {{
    0%   {{ transform: translateY(0) scale(1); }}
    30%  {{ transform: translateY(-10px) scale(1.1); }}
    60%  {{ transform: translateY(5px) scale(0.95); }}
    100% {{ transform: translateY(0) scale(1); }}
}}

.emotion-name {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #a8d0e6;
    letter-spacing: 1px;
}}

/* Instructions Section - Improved Compact Version */
.instructions-section {{
    margin: 6rem 0;  /* เพิ่มระยะห่างขอบบน-ล่าง */
    animation: fadeInUp 0.8s ease-out;
}}

.instructions-container {{
     background: linear-gradient(90deg, #7db3d3 0%, #a8d0e6 100%);
    padding: 2.5rem;
    border-radius: 20px;
    border: 1px solid rgba(148, 163, 184, 0.2);
    text-align: center;
    position: relative;
    box-shadow: 
        0 20px 40px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}}

.instructions-container::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 50% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
    border-radius: 20px;
    pointer-events: none;
}}

.instructions-container::after {{
   content: '';
    position: absolute;
    top: -1px;
    left: -1px;
    right: -1px;
    bottom: -1px;
    background: linear-gradient(
        45deg, 
        rgba(59, 130, 246, 0.3) 0%, 
        rgba(139, 92, 246, 0.3) 33%, 
        rgba(16, 185, 129, 0.3) 66%, 
        rgba(59, 130, 246, 0.3) 100%
    );
    border-radius: 20px;
    z-index: -1;
    animation: borderGlow 3s ease-in-out infinite;
}}

.section-title {{
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff !important;  /* ขาวสด */
    margin: 6rem 0 3rem 0;  /* เพิ่มเว้นบน-ล่าง */
    position: relative;
    text-align: center;
    letter-spacing: 1px;

    /* ยกเลิกทุก effect เดิมที่ทำให้มันไม่ขาว */
    background: none !important;
    -webkit-background-clip: unset !important;
    -webkit-text-fill-color: #ffffff !important; 
    background-clip: unset !important;
}}

@keyframes borderGlow {{
    0%, 100% {{ opacity: 0.5; }} /* เริ่มและจบครึ่งโปร่งใส */
    50% {{ opacity: 1; }}        /* กลาง animation เต็มความชัด */
}}

.section-title::after {{
    content: "";
    display: block;
    width: 80px; /* ความยาวเส้น */
    height: 4px; /* ความหนาเส้น */
    margin: 0.6rem auto 0; /* จัดให้อยู่กลาง */
    border-radius: 2px;
    background: linear-gradient(to right, #ffffff, #a1a1a1); /* ไล่สี ขาว → เทา */
}}


.instructions-steps {{
    display: flex;
    justify-content: center;
    align-items: stretch;
    gap: 2rem;
    position: relative;
    z-index: 1;
}}

.step-item {{
    background: rgba(255, 255, 255, 0.08);
    padding: 2rem 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    flex: 1;
    max-width: 280px;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
    opacity: 0;
    transform: translateY(30px);
}}

.step-item:nth-child(1) {{
    animation: slideInStep 0.8s ease-out 0.2s forwards;
}}

.step-item:nth-child(2) {{
    animation: slideInStep 0.8s ease-out 0.4s forwards;
}}

.step-item:nth-child(3) {{
    animation: slideInStep 0.8s ease-out 0.6s forwards;
}}

.step-item::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
        135deg,
        rgba(59, 130, 246, 0.1) 0%,
        rgba(139, 92, 246, 0.1) 50%,
        rgba(16, 185, 129, 0.1) 100%
    );
    opacity: 0;
    transition: opacity 0.4s ease;
    border-radius: 16px;
}}

.step-item:hover {{
    transform: translateY(-8px) scale(1.02);
    background: rgba(255, 255, 255, 0.12);
    box-shadow: 
        0 20px 40px rgba(0, 0, 0, 0.3),
        0 0 30px rgba(59, 130, 246, 0.2);
    border-color: rgba(255, 255, 255, 0.25);
}}

.step-item:hover::before {{
    opacity: 1;
}}

.step-number {{
    width: 50px;
    height: 50px;
    background: linear-gradient(135deg, #3b82f6, #1d4ed8, #7c3aed);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.2rem;
    margin: 0 auto 1.2rem;
    box-shadow: 
        0 8px 20px rgba(59, 130, 246, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    position: relative;
    z-index: 2;
    transition: all 0.3s ease;
}}

.step-item:hover .step-number {{
    transform: scale(1.1) rotate(5deg);
    box-shadow: 
        0 12px 30px rgba(59, 130, 246, 0.6),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
}}

.step-text {{
    color: #e2e8f0;
    font-size: 0.85rem;
    line-height: 1.4;
    margin: 0;
    font-weight: 400;
}}

/* Animations */
@keyframes slideInStep {{
    from {{
        opacity: 0;
        transform: translateY(30px) scale(0.95);
    }}
    to {{
        opacity: 1;
        transform: translateY(0) scale(1);
    }}
}}

/* Responsive Design */
@media (max-width: 768px) {{
    .instructions-steps {{
        flex-direction: column;
        gap: 1.5rem;
    }}
    
    .step-item {{
        max-width: none;
        padding: 1.5rem;
    }}
    
    .instructions-container {{
        padding: 1.5rem;
    }}
    
    .section-title {{
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
    }}
    
    .step-text {{
        font-size: 0.9rem;
    }}
    
    .step-number {{
        width: 45px;
        height: 45px;
        font-size: 1.1rem;
    }}
}}

@media (max-width: 480px) {{
    .step-item {{
        padding: 1.2rem;
    }}
    
    .step-number {{
        width: 40px;
        height: 40px;
        font-size: 1rem;
    }}
    
    .step-text {{
        font-size: 0.85rem;
    }}
    
    .instructions-container {{
        padding: 1.2rem;
    }}
    
    .section-title {{
        font-size: 1.5rem;
    }}
}}
.upload-section {{
    background: linear-gradient(135deg, #1a1a1a, #0a0a0a);
    padding: 2.5rem;
    border-radius: 20px;
    border: 2px dashed #666;
    text-align: center;
    margin: 2rem 0;
    transition: all 0.4s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}}

.upload-section:hover {{
    border-color: #999;
    background: linear-gradient(135deg, #2d2d2d, #1a1a1a);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6);
}}

.prediction-card {{
    background: linear-gradient(135deg, #1a1a1a, #0a0a0a);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    margin: 1rem 0;
    border: 1px solid #333;
    transition: all 0.3s ease;
}}

.prediction-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
    border-color: #666;
}}

.emotion-result {{
    padding: 1.2rem;
    border-radius: 15px;
    margin: 0.8rem 0;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}}

.emotion-result:hover {{
    transform: translateX(8px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}}

.emotion-happy {{ 
    background: linear-gradient(135deg, #2d4a2d, #4a6b4a);
    box-shadow: 0 4px 15px rgba(45, 74, 45, 0.3);
}}
.emotion-sad {{ 
    background: linear-gradient(135deg, #2d3a4a, #4a5a6b);
    box-shadow: 0 4px 15px rgba(45, 58, 74, 0.3);
}}
.emotion-fear {{ 
    background: linear-gradient(135deg, #4a2d2d, #6b4a4a);
    box-shadow: 0 4px 15px rgba(74, 45, 45, 0.3);
}}
.emotion-neutral {{ 
    background: linear-gradient(135deg, #3a3a3a, #4a4a4a);
    box-shadow: 0 4px 15px rgba(58, 58, 58, 0.3);
}}

.stButton > button {{
    background: linear-gradient(135deg, #666 0%, #333 100%);
    color: white;
    border: none;
    border-radius: 30px;
    padding: 1rem 2.5rem;
    font-size: 1.2rem;
    font-weight: 600;
    transition: all 0.4s ease;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    font-family: 'Poppins', sans-serif;
}}

.stButton > button:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
    background: linear-gradient(135deg, #333 0%, #666 100%);
}}

.info-box {{
    background: linear-gradient(135deg, #333, #000);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px);
    border: 1px solid #666;
}}

h1, h2, h3 {{
    font-family: 'Poppins', sans-serif;
    color: #ffffff !important;
}}

.stMarkdown p {{
    font-family: 'Inter', sans-serif;
    color: #cccccc !important;
}}

/* Streamlit elements dark theme */
.stSelectbox > div > div {{
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border-color: #666 !important;
}}

.stTextInput > div > div > input {{
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border-color: #666 !important;
}}

.stFileUploader > div {{
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border-color: #666 !important;
}}

.uploadedFile {{
    background-color: #2d2d2d !important;
    color: #ffffff !important;
}}

/* Progress bar */
.stProgress > div > div > div {{
    background-color: #666 !important;
}}

/* Metrics */
.metric-container {{
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #666 !important;
}}

/* Responsive Design */
@media (max-width: 768px) {{
    .about-container {{
        grid-template-columns: 1fr;
        gap: 2rem;
    }}
    
    .about-title {{
        font-size: 2.2rem;
    }}
    
    .emotions-grid {{
        grid-template-columns: 1fr;
    }}
    
    .instructions-steps {{
        grid-template-columns: 1fr;
    }}
    
    .banner-section {{
        height: 350px;
    }}
    
    .banner-text {{
        max-width: 90%;
    }}
    
    .banner-text h1 {{
        font-size: 2.5rem;
    }}
    
    .banner-text h2 {{
        font-size: 1.5rem;
    }}
    
    .banner-text p {{
        font-size: 1rem;
        max-width: 100%;
    }}
    
    .banner-content {{
        padding: 0 5px;
    }}
    
    .logo-content {{
        font-size: 1.1rem;
    }}
    
    .brain-icon {{
    font-size: 1.8rem;
    color: #fff !important;
}}

.emotions-title::after {{
    background: #2146a0;
}}

.info-box {{
    border: 1px solid #2146a0;
}}

@media (max-width: 480px) {{
    .banner-section {{
        height: 300px;
    }}
    
    .banner-text h1 {{
        font-size: 2rem;
    }}
    
    .banner-text h2 {{
        font-size: 1.3rem;
    }}
    
    .banner-content {{
        padding: 0 10px;
    }}
    
    .about-section {{
        padding: 1.5rem;
    }}
}}
</style>
"""

# Apply CSS
st.markdown(create_css_with_banner(), unsafe_allow_html=True)

# Logo Bar with FontAwesome CDN
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<div class="logo-bar">
    <div class="logo-content">
        <i class="fas fa-brain brain-icon"></i>
        <span style="color:#fff;">AI EMOTION DETECTION SYSTEM</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Banner Section
st.markdown("""
<div class="banner-section">
    <div class="banner-overlay"></div>
    <div class="banner-content">
        <div class="banner-text">
            <h1>AI EMOTION DETECTION</h1>
            <h2>ADVANCED DEEPLEARNING TECHNOLOGY</h2>
            <p>Harness the power of artificial intelligence to analyze and understand human emotions with unprecedented accuracy and precision.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# About Section
st.markdown("""
<div class="about-section">
    <div class="about-container">
        <div class="about-main" style="text-align: center; grid-column: 1 / -1; justify-self: center;">
            <h2 class="about-title">About This App</h2>
            <p class="about-description">
               Welcome to our cutting-edge AI emotion detection system.
This advanced application utilizes deep learning technology to analyze facial expressions and detect emotions with high accuracy.
Beyond just recognition, our system provides real-time feedback, enabling seamless integration into healthcare, education, and customer experience platforms.
With continuous learning and adaptability, it ensures reliable performance across diverse environments and user groups.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Emotions Section
st.markdown("""
<div class="emotions-sidebar">
    <div class="emotions-container">
       <h2 class="about-title">Supported Emotions</h2>
        <div class="emotions-grid">
            <div class="emotion-item">
                <span class="emotion-emoji">😨</span>
                <span class="emotion-name">Fear</span>
            </div>
            <div class="emotion-item">
                <span class="emotion-emoji">😊</span>
                <span class="emotion-name">Happy</span>
            </div>
            <div class="emotion-item">
                <span class="emotion-emoji">😐</span>
                <span class="emotion-name">Neutral</span>
            </div>
            <div class="emotion-item">
                <span class="emotion-emoji">😢</span>
                <span class="emotion-name">Sad</span><br>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Instructions Section
st.markdown("""
<div class="instructions-section"><br>
        <h2 class="section-title">How to Use</h2>
        <div class="instructions-steps">
            <div class="step-item">
                <div class="step-number">1</div>
                <p class="step-text">Upload an image (JPG, JPEG, PNG format)</p>
            </div>
            <div class="step-item">
                <div class="step-number">2</div>
                <p class="step-text">Click 'Analyze Emotion' button</p>
            </div>
            <div class="step-item">
                <div class="step-number">3</div>
                <p class="step-text">View detailed prediction results</p><br>
            </div>
        </div>
</div>
""", unsafe_allow_html=True)

# กำหนดชื่อคลาสอารมณ์ที่โมเดลสามารถทำนายได้
class_names = ["Fear", "Happy", "Neutral", "Sad"]

# ฟังก์ชันทำนายที่แก้ไขแล้ว
def pred_class(model, image, class_names, device='cpu'):
    """
    ฟังก์ชันทำนายอารมณ์จากภาพ
    
    Args:
        model: โมเดล PyTorch ที่โหลดแล้ว
        image: PIL Image object
        class_names: รายชื่อคลาสอารมณ์
        device: device ที่ใช้ ('cpu' หรือ 'cuda')
    
    Returns:
        tuple: (predicted_class_name, confidence_score, all_probabilities)
    """
    try:
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Transform image
        img_tensor = transform(image).unsqueeze(0).to(device)
        img_tensor = img_tensor.float()
        
        # Prediction
        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
            
        # Get all probabilities
        all_probs = probabilities[0].cpu().numpy()
        predicted_class = class_names[predicted_idx]
        
        return predicted_class, confidence, all_probs
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, 0.0, None

# Load Model
@st.cache_resource
def load_model():

    model_path = "efficientnet_b3_checkpoint_fold1.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    # ถ้าไฟล์ไม่มี ให้ดาวน์โหลดก่อน
    if not os.path.exists(model_path):
        try:
            url = "https://drive.google.com/uc?id=1TUVnEHkl3fd-5olrDR-wTlkGFKakAIaB"
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None, device
        
    # ✅ โหลด checkpoint แบบ allow Lightning class
    try:
        with safe_globals([lightning.fabric.wrappers._FabricModule]):
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device
        
    # ดึง state_dict จาก checkpoint
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # fallback: ถ้าโหลดได้เป็น object ให้ดึง state_dict() โดยตรง
        try:
            state_dict = ckpt.state_dict()
        except Exception:
            st.error("Checkpoint format not supported.{e}")
            return None, device
    
    # สร้างโมเดล
    model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=4)
    
    # map key ให้ตรง (ลบ prefix 'model.' ที่ Lightning ชอบใส่)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("model.", "")
        new_state_dict[new_key] = v
    
    # โหลดน้ำหนัก
    try:
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device
       

# เรียกใช้
model, device = load_model()

# Main Content Area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div style="margin-bottom: 1.2rem; text-align: left;"><br><br>
        <h3 style="
            color: #e0e0e0; 
            font-size: 1.5rem; 
            font-weight: 700; 
            margin-bottom: 0.3rem;
            display: flex;
            align-items: center;
            gap: 10px;
            letter-spacing: 1px;
        ">
            <i class="fas fa-upload" style="color:#ffffff; font-size:1.6rem;"></i>
            <span>Image Upload</span>
        </h3>
        <div style="
            width: 60px; 
            height: 3px; 
            background: linear-gradient(90deg, #7db3d3 0%, #a8d0e6 100%);
            border-radius: 2px;
            margin-top: 0.2rem;
            margin-bottom: 0.5rem;
        "></div>
    </div>
    """, unsafe_allow_html=True)
    
    # สร้างตัวแปร uploaded_image สำหรับการอัปโหลดไฟล์
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    # แสดงภาพเฉพาะในคอลัมน์แรก
    image = None
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        #st.image(image, caption="Uploaded Image", use_container_width=True)
        st.image(image, caption="Uploaded Image", width = 650, height = 500, use_container_width=False)

with col2:
    st.markdown("""
    <div style="margin-bottom: 1.2rem; text-align: left;"><br><br>
        <h3 style="
            color: #e0e0e0; 
            font-size: 1.5rem; 
            font-weight: 700; 
            margin-bottom: 0.3rem;
            display: flex;
            align-items: center;
            gap: 10px;
            letter-spacing: 1px;
        ">
            <i class="fas fa-image" style="color:#ffffff; font-size:1.6rem;"></i>
            <span>Image Preview</span>
        </h3>
        <div style="
            width: 60px; 
            height: 3px; 
            background: linear-gradient(90deg, #7db3d3 0%, #a8d0e6 100%);
            border-radius: 2px;
            margin-top: 0.2rem;
            margin-bottom: 0.5rem;
        "></div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_image is not None and image is not None:
        #st.image(uploaded_image, width='stretch')
        file_type = getattr(uploaded_image, 'type', 'unknown')
        file_size_kb = len(uploaded_image.getvalue()) / 1024
        st.markdown(f"""
        <div class="info-box" style="
            background: linear-gradient(135deg, #181c20 60%, #232b36 100%);
            border-radius: 12px;
            border: 1px solid #333;
            padding: 1rem;
            margin-top: 1rem;
            color: #e0e0e0;
        ">
            <strong>Image Details:</strong><br>
            <span style="color:#1F425D;">Size:</span> {image.size[0]} x {image.size[1]} pixels<br>
            <span style="color:#1F425D;">Format:</span> {file_type}<br>
            <span style="color:#1F425D;">File size:</span> {file_size_kb:.1f} KB
        </div>
        """, unsafe_allow_html=True)
        
        # ปุ่มสำหรับวิเคราะห์อารมณ์
        st.markdown("<br>", unsafe_allow_html=True)
        # ใช้ session state เพื่อจัดการการแสดงผล
        if 'prediction_done' not in st.session_state:
            st.session_state.prediction_done = False
            
        if st.button("Analyze Emotion", type="primary", width='stretch',use_container_width=True):
            with st.spinner("Analyzing emotions..."):
                if model is not None:
                    try:
                        # เรียกใช้ฟังก์ชันทำนาย
                        predicted_class, confidence, all_probs = pred_class(model, image, class_names, device)
                        
                        if predicted_class is not None:
                            # เก็บผลลัพธ์ใน session state
                            st.session_state.prediction_result = {
                                'predicted_class': predicted_class,
                                'confidence': confidence,
                                'all_probs': all_probs
                            }
                            st.session_state.prediction_done = True
                            st.success(f"Analysis completed! Predicted: {predicted_class}")
                            st.info(f"Confidence: {confidence*100:.1f}%")
                        else:
                            st.error("Failed to analyze emotion")
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                else:
                    st.error("Model not loaded properly")
    else:
        st.markdown("""
        <div style="
            text-align: center; 
            padding: 2.5rem 1rem; 
            color: #7db3d3; 
            background: linear-gradient(135deg, #181c20 60%, #232b36 100%); 
            border-radius: 16px;
            border: 1px solid #333;
            margin-top: 1rem;
        ">
            <i class="fas fa-image" style="font-size:2.5rem; color:#7db3d3; margin-bottom:0.5rem;"></i>
            <h4 style="color:#e0e0e0; margin:0 0 0.5rem 0;">No Image Selected</h4>
            <p style="color:#b0b8c1; margin:0;">Please upload an image to see the preview.</p>
        </div>
        """, unsafe_allow_html=True)


# Initialize session state
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
    
# Prediction Section
if uploaded_image is not None and model is not None and st.session_state.prediction_done and st.session_state.prediction_result is not None:
    st.markdown("---")
    
    # ดึงผลลัพธ์จาก session state
    result = st.session_state.prediction_result
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    all_probs = result['all_probs']
    
    emoji_map = {'Fear': '😨', 'Happy': '😊', 'Neutral': '😐', 'Sad': '😢'}
    color_map = {'Fear': 'emotion-fear', 'Happy': 'emotion-happy',
                 'Neutral': 'emotion-neutral', 'Sad': 'emotion-sad'}
    
    # Results Section
    st.markdown("## Prediction Results")
    
    # Create two columns for results
    result_col1, result_col2 = st.columns([2, 1])

    with result_col1:
        #st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        max_index = np.argmax(all_probs)
        
        # Display results with styling
        for i, (emotion, prob) in enumerate(zip(class_names, all_probs)):
            emoji = emoji_map[emotion]
            percentage = prob * 100
            is_max = (i == max_index)

            # Create styled result
            if is_max:
                st.markdown(f"""
                <div class="emotion-result {color_map[emotion]}" style="border: 3px solid gold;">
                    <h3 style="margin:0; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                        {emoji} <strong>{emotion}</strong>: {percentage:.1f}%
                    </h3>
                    <p style="margin:0; color: white; font-size: 0.9em;">Primary Detection</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.3rem 0; background: rgba(255,255,255,0.1); border-radius: 8px; border-left: 4px solid #5897c2;">
                    <span style="font-size: 1.1em; color: #e0e0e0;">{emoji} {emotion}: <strong>{percentage:.1f}%</strong></span>
                </div>
                """, unsafe_allow_html=True)

        #st.markdown('</div>', unsafe_allow_html=True)

    with result_col2:
        # Create a donut chart with brand colors
        fig = go.Figure(data=[go.Pie(
            labels=[f"{emoji_map[emotion]} {emotion}" for emotion in class_names],
            values=[prob * 100 for prob in all_probs],
            hole=.3,
            marker_colors=["#254e94", '#4caf50', '#607d8b', '#5897c2']
        )])

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            title="Emotion Distribution",
            annotations=[dict(text='Confidence', x=0.5, y=0.5, font_size=16, showarrow=False)],
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        st.plotly_chart(fig, use_container_width=True)
        
    # Confidence indicator
    max_confidence = all_probs[max_index] * 100
    if max_confidence > 80:
        confidence_color = "#4caf50"
        confidence_text = "High Confidence"
    elif max_confidence > 60:
        confidence_color = "#5897c2"
        confidence_text = "Medium Confidence"
    else:
        confidence_color = "#f44336"
        confidence_text = "Low Confidence"

    st.markdown(f"""
    <div style="text-align: center; margin: 2rem 0;">
        <span style="background: {confidence_color}; color: white; padding: 0.5rem 1rem; 
        border-radius: 20px; font-weight: bold;">
            🎯 {confidence_text}: {max_confidence:.1f}%
        </span>
    </div>
    """, unsafe_allow_html=True)

# Footer with enhanced styling
st.markdown("---")
st.markdown("""
<div style="
    text-align: center; 
    color: #ffffff; 
    padding: 3rem 2rem; 
    background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #112e63 100%); 
    border-radius: 25px; 
    margin-top: 3rem;
    position: relative;
    overflow: hidden;
">
    <div style="position: relative; z-index: 1;">
        <h3 style="color: #ffffff; margin-bottom: 1rem; font-size: 1.5rem;">🤖 AI Emotion Detection System</h3>
        <p style="color: #a8d0e6; margin-bottom: 1rem; font-size: 1.1rem;">
            Powered by <strong>Deep Learning</strong> | Built with ❤️ using <strong>Streamlit</strong>
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;">
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; backdrop-filter: blur(10px);">
                <strong>⚡ Fast Analysis</strong><br>
                <small>Real-time processing</small>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; backdrop-filter: blur(10px);">
                <strong>🎯 High Accuracy</strong><br>
                <small>ResNet-50 powered</small>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; backdrop-filter: blur(10px);">
                <strong>🔒 Secure</strong><br>
                <small>No data storage</small>
            </div>
        </div>
        <p style="color: #7db3d3; margin-top: 2rem; font-size: 0.9rem;">
            💡 <em>For optimal results, use clear images with visible facial expressions</em>
        </p>
    </div>
    <div style="
        position: absolute; 
        top: -50%; 
        left: -50%; 
        width: 200%; 
        height: 200%; 
        background: linear-gradient(135deg, #000000 80%, #1a1a1a 100%, #112e63 100%);
        background-size: 100px 100px;
        animation: float 15s ease-in-out infinite;
    "></div>
</div>
""", unsafe_allow_html=True)