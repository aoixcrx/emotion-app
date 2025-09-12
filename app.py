import os
import streamlit as st
import torch
from PIL import Image
from prediction import pred_class
import numpy as np
import plotly.graph_objects as go
import gdown

# Page Configuration
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        color: #f0f2f6;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    .upload-section:hover {
        border-color: #764ba2;
        background: #f0f2f6;
    }
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .emotion-result {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .emotion-result:hover {
        transform: translateX(5px);
    }
    .emotion-happy { background: linear-gradient(90deg, #56ab2f, #a8e6cf); }
    .emotion-sad { background: linear-gradient(90deg, #4a90e2, #7bb3f0); }
    .emotion-fear { background: linear-gradient(90deg, #ff6b6b, #ffa8a8); }
    .emotion-neutral { background: linear-gradient(90deg, #95a5a6, #bdc3c7); }

    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .info-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-header">
    <h1> AI Emotion Detection</h1>
    <p class="subtitle">Upload an image and let our AI identify the emotion with advanced deep learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.markdown("### About This App")
    st.markdown("""
    This application uses a **EfficientNet-B3** deep learning model to classify emotions in images.

    **Supported Emotions:**
    - 😨 Fear
    - 😊 Happy  
    - 😐 Neutral
    - 😢 Sad
    """)
    st.markdown("---")
    st.info("Using EfficientNet-B3 architecture trained on emotion dataset")

# Load Model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "efficientnet_b3_checkpoint_fold1.pt"

    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1TUVnEHkl3fd-5olrDR-wTlkGFKakAIaB"
        gdown.download(url, model_path, quiet=False)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            from timm import create_model
            model = create_model("efficientnet_b3", pretrained=False, num_classes=4)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = checkpoint
        model = model.to(torch.float32).eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

model, device = load_model()

# Upload + Preview
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### Image Upload")
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_image = st.file_uploader(
    'Drop your image here or click to browse',
    type=['jpg', 'jpeg', 'png'],
    help="Supported formats: JPG, JPEG, PNG"
)
st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### Image Preview")
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption=uploaded_image.name, use_container_width=True)

# Prediction
if uploaded_image is not None and model is not None:
    st.markdown("---")
    predict_button = st.button("Analyze Emotion", use_container_width=True)
    if predict_button:
        with st.spinner("Analyzing emotion..."):
            try:
                class_names = ["Fear", "Happy", "Neutral", "Sad"]
                emoji_map = {'Fear': '😨', 'Happy': '😊', 'Neutral': '😐', 'Sad': '😢'}
                color_map = {'Fear': 'emotion-fear', 'Happy': 'emotion-happy',
                             'Neutral': 'emotion-neutral', 'Sad': 'emotion-sad'}

                probs, classname = pred_class(model, image, class_names)
                max_index = np.argmax(probs)

                # Show results
                st.subheader("Prediction Results")
                for i, (emotion, prob) in enumerate(zip(class_names, probs)):
                    emoji = emoji_map[emotion]
                    if i == max_index:
                        st.success(f"{emoji} {emotion}: {prob*100:.1f}% (Primary Detection)")
                    else:
                        st.write(f"{emoji} {emotion}: {prob*100:.1f}%")

                # Donut Chart
                fig = go.Figure(data=[go.Pie(
                    labels=[f"{emoji_map[e]} {e}" for e in class_names],
                    values=[p*100 for p in probs],
                    hole=.3
                )])
                fig.update_layout(title="Emotion Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error during prediction: {e}")


# import os
# import streamlit as st
# import torch
# from PIL import Image
# from prediction import pred_class
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import gdown

# # Page Configuration
# st.set_page_config(
#     page_title="Emotion Detection AI",
#     page_icon="😊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for beautiful styling
# st.markdown("""
# <style>
#     .main-header {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#         border-radius: 15px;
#         text-align: center;
#         margin-bottom: 2rem;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.1);
#     }
#     .main-header h1 {
#         color: white;
#         font-size: 3rem;
#         margin: 0;
#         font-weight: 700;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#     }
#     .subtitle {
#         color: #f0f2f6;
#         font-size: 1.2rem;
#         margin-top: 0.5rem;
#     }
#     .upload-section {
#         background: #f8f9fa;
#         padding: 2rem;
#         border-radius: 15px;
#         border: 2px dashed #667eea;
#         text-align: center;
#         margin: 2rem 0;
#         transition: all 0.3s ease;
#     }
#     .upload-section:hover {
#         border-color: #764ba2;
#         background: #f0f2f6;
#     }
#     .prediction-card {
#         background: white;
#         padding: 1.5rem;
#         border-radius: 15px;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.1);
#         margin: 1rem 0;
#         border-left: 5px solid #667eea;
#     }
#     .emotion-result {
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 0.5rem 0;
#         transition: all 0.3s ease;
#     }
#     .emotion-result:hover {
#         transform: translateX(5px);
#     }
#     .emotion-happy { background: linear-gradient(90deg, #56ab2f, #a8e6cf); }
#     .emotion-sad { background: linear-gradient(90deg, #4a90e2, #7bb3f0); }
#     .emotion-fear { background: linear-gradient(90deg, #ff6b6b, #ffa8a8); }
#     .emotion-neutral { background: linear-gradient(90deg, #95a5a6, #bdc3c7); }

#     .stButton > button {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         border-radius: 25px;
#         padding: 0.75rem 2rem;
#         font-size: 1.1rem;
#         font-weight: 600;
#         transition: all 0.3s ease;
#         box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
#     }
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
#     }
#     .info-box {
#         background: linear-gradient(135deg, #667eea, #764ba2);
#         color: white;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Header Section
# st.markdown("""
# <div class="main-header">
#     <h1> AI Emotion Detection</h1>
#     <p class="subtitle">Upload an image and let our AI identify the emotion with advanced deep learning</p>
# </div>
# """, unsafe_allow_html=True)

# # Sidebar with information
# with st.sidebar:
#     st.markdown("### About This App")
#     st.markdown("""
#     This application uses a **ResNet-50** deep learning model to classify emotions in images.

#     **Supported Emotions:**
#     - 😨 Fear
#     - 😊 Happy  
#     - 😐 Neutral
#     - 😢 Sad

#     **Instructions:**
#     1. Upload an image (JPG, JPEG, PNG)
#     2. Click 'Analyze Emotion'
#     3. View the prediction results
#     """)

#     st.markdown("---")
#     st.markdown("### Model Info")
#     st.info("Using ResNet-50 architecture trained on emotion dataset")


# # Load Model
# @st.cache_resource
# def load_model():
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model_path = "efficientnet_b3_checkpoint_fold1.pt"

#     # ถ้าไฟล์ไม่มีให้โหลดจาก Google Drive
#     if not os.path.exists(model_path):
#         url = "https://drive.google.com/uc?id=1TUVnEHkl3fd-5olrDR-wTlkGFKakAIaB"
#         gdown.download(url, model_path, quiet=False)
#         try:
#             model = torch.load('efficientnet_b3_checkpoint_fold1.pt', map_location=device, weights_only=False)
#             model = model.to(torch.float32)   # 🔥 บังคับทุก layer/param เป็น float32
#             model.eval()
#             return model, device
#         except:
#             st.error("Model file not found! Please check the path.")
#             return None, device
#     # โหลดโมเดล
#     try:
#         model = torch.load(model_path, map_location=device, weights_only=False)
#         model = model.to(torch.float32)   # 🔥 บังคับทุก layer/param เป็น float32
#         model.eval()
#         return model, device
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None, device


# model, device = load_model()

# # Main Content Area
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.markdown("### Image Upload")

# # File Upload Section
# st.markdown('<div class="upload-section">', unsafe_allow_html=True)
# uploaded_image = st.file_uploader(
#     'Drop your image here or click to browse',
#     type=['jpg', 'jpeg', 'png'],
#     help="Supported formats: JPG, JPEG, PNG"
# )
# st.markdown('</div>', unsafe_allow_html=True)

# with col2:
#     st.markdown("### Image Preview")

#     if uploaded_image is not None:
#         image = Image.open(uploaded_image).convert('RGB')
#         st.image(
#             image,
#             caption=f'{uploaded_image.name}',
#             use_container_width=True
#         )

#         # Image info
#         st.markdown(f"""
#         <div class="info-box">
#             <strong>Image Details:</strong><br>
#             Size : {image.size[0]} x {image.size[1]} pixels<br>
#             Format : {uploaded_image.type}<br>
#             File size : {len(uploaded_image.getvalue()) / 1024:.1f} KB
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown("""
#         <div style="text-align: center; padding: 3rem; color: #666;">
#             <h3>No Image Selected</h3>
#             <p>Please upload an image to see the preview</p>
#         </div>
#         """, unsafe_allow_html=True)

# # Prediction Section
# if uploaded_image is not None and model is not None:
#     st.markdown("---")

#     col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
#     with col_btn2:
#         predict_button = st.button('Analyze Emotion', use_container_width=True)

#     if predict_button:
#         with st.spinner('Analyzing emotion... Please wait'):
#             try:
#                 class_name = ['Fear', 'Happy', 'Neutral', 'Sad']
#                 emoji_map = {'Fear': '😨', 'Happy': '😊', 'Neutral': '😐', 'Sad': '😢'}
#                 color_map = {'Fear': 'emotion-fear', 'Happy': 'emotion-happy',
#                              'Neutral': 'emotion-neutral', 'Sad': 'emotion-sad'}

#                 # Get prediction
#                 probli = pred_class(model, image, class_name)
#                 max_index = np.argmax(probli[0])

#                 # Results Section
#                 st.markdown("## Prediction Results")

#                 # Create two columns for results
#                 result_col1, result_col2 = st.columns([2, 1])

#                 with result_col1:
#                     st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

#                     # Display results with styling
#                     for i, (emotion, prob) in enumerate(zip(class_name, probli[0])):
#                         emoji = emoji_map[emotion]
#                         percentage = prob * 100
#                         is_max = (i == max_index)

#                         # Create styled result
#                         if is_max:
#                             st.markdown(f"""
#                             <div class="emotion-result {color_map[emotion]}" style="border: 3px solid #gold;">
#                                 <h3 style="margin:0; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
#                                     {emoji} <strong>{emotion}</strong>: {percentage:.1f}%
#                                 </h3>
#                                 <p style="margin:0; color: white; font-size: 0.9em;">Primary Detection</p>
#                             </div>
#                             """, unsafe_allow_html=True)
#                         else:
#                             st.markdown(f"""
#                             <div style="padding: 0.5rem; margin: 0.3rem 0; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #ddd;">
#                                 <span style="font-size: 1.1em;">{emoji} {emotion}: <strong>{percentage:.1f}%</strong></span>
#                             </div>
#                             """, unsafe_allow_html=True)

#                     st.markdown('</div>', unsafe_allow_html=True)

#                 with result_col2:
#                     # Create a donut chart
#                     fig = go.Figure(data=[go.Pie(
#                         labels=[f"{emoji_map[emotion]} {emotion}" for emotion in class_name],
#                         values=[prob * 100 for prob in probli[0]],
#                         hole=.3,
#                         marker_colors=['#ff6b6b', '#56ab2f', '#95a5a6', '#4a90e2']
#                     )])

#                     fig.update_traces(textposition='inside', textinfo='percent+label')
#                     fig.update_layout(
#                         title="Emotion Distribution",
#                         annotations=[dict(text='Confidence', x=0.5, y=0.5, font_size=16, showarrow=False)],
#                         height=400,
#                         showlegend=False
#                     )

#                     st.plotly_chart(fig, use_container_width=True)

#                 # Confidence indicator
#                 max_confidence = probli[0][max_index] * 100
#                 if max_confidence > 80:
#                     confidence_color = "green"
#                     confidence_text = "High Confidence"
#                 elif max_confidence > 60:
#                     confidence_color = "orange"
#                     confidence_text = "Medium Confidence"
#                 else:
#                     confidence_color = "red"
#                     confidence_text = "Low Confidence"

#                 st.markdown(f"""
#                 <div style="text-align: center; margin: 2rem 0;">
#                     <span style="background: {confidence_color}; color: white; padding: 0.5rem 1rem; 
#                     border-radius: 20px; font-weight: bold;">
#                         {confidence_text}: {max_confidence:.1f}%
#                     </span>
#                 </div>
#                 """, unsafe_allow_html=True)

#             except Exception as e:
#                 st.error(f"Error during prediction: {str(e)}")
#                 st.info("Please make sure the model file exists and the prediction function works correctly.")

# # Footer
# st.markdown("---")
# st.markdown("""
# <div style="text-align: center; color: #666; padding: 2rem;">
#     <p>Powered by <strong>Deep Learning</strong> | Built with using <strong>Streamlit</strong></p>
#     <p><small>For best results, use clear images with visible faces</small></p>
# </div>

# """, unsafe_allow_html=True)








