import streamlit as st
import torch
from PIL import Image
from prediction import pred_class
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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

# Sidebar with information
with st.sidebar:
    st.markdown("### About This App")
    st.markdown("""
    This application uses a **ResNet-50** deep learning model to classify emotions in images.

    **Supported Emotions:**
    - 😨 Fear
    - 😊 Happy  
    - 😐 Neutral
    - 😢 Sad

    **Instructions:**
    1. Upload an image (JPG, JPEG, PNG)
    2. Click 'Analyze Emotion'
    3. View the prediction results
    """)

    st.markdown("---")
    st.markdown("### Model Info")
    st.info("Using ResNet-50 architecture trained on emotion dataset")

# Main Content Area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Image Upload")


    # # Load Model
    # @st.cache_resource
    # def load_model():
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     try:
    #         model = torch.load('efficientnet_b3_checkpoint_fold1.pt',map_location=device,weights_only=False)
    #         return model, device
    #     except:
    #         st.error("Model file not found! Please check the path.")
    #         return None, device


    # model, device = load_model()

    # Load Model
   @st.cache_resource
def load_model(model_name='efficientnet_b3_checkpoint_fold1.pt', debug=False):
    """
    Load PyTorch model with comprehensive error handling and debugging
    
    Args:
        model_name (str): Name of the model file to load
        debug (bool): Whether to show detailed debugging information
    
    Returns:
        tuple: (model, device) where model is the loaded PyTorch model and device is the torch device
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = Path(model_name)
    
    if debug:
        st.write(f"🔍 Looking for model at: {model_path.absolute()}")
        st.write(f"📁 Current directory: {Path.cwd()}")
        st.write(f"🖥️ Using device: {device}")
    
    # List all files in current directory
    current_dir = Path(".")
    all_files = list(current_dir.iterdir())
    pt_files = [f for f in all_files if f.suffix == '.pt']
    
    if debug:
        st.write("📋 Files in current directory:")
        for file in all_files:
            if file.suffix == '.pt':
                try:
                    file_size = file.stat().st_size / (1024*1024)  # MB
                    st.write(f"  ✅ {file.name} ({file_size:.2f} MB)")
                except OSError:
                    st.write(f"  ⚠️ {file.name} (size unknown)")
            elif file.is_file():
                st.write(f"  📄 {file.name}")
    
    # Check if the specified model file exists
    if not model_path.exists():
        st.error(f"❌ Model file '{model_name}' not found!")
        
        if pt_files:
            st.warning(f"🔧 Found these .pt files instead: {[f.name for f in pt_files]}")
            # Use the first .pt file found
            model_path = pt_files[0]
            st.info(f"🔄 Trying to use: {model_path.name}")
        else:
            st.error("🚫 No .pt files found in directory!")
            st.info("💡 Make sure to upload your model file to the app directory")
            return None, device
    
    try:
        st.info(f"📥 Loading model from: {model_path.name}")
        
        # Load model with progress indicator
        with st.spinner("Loading model..."):
            # Check file size for large models
            file_size_mb = model_path.stat().st_size / (1024*1024)
            if file_size_mb > 100:
                st.warning(f"⏳ Large model detected ({file_size_mb:.1f} MB). This may take a while...")
            
            model = torch.load(model_path, map_location=device, weights_only=False)
        
        # Validate model structure
        if hasattr(model, 'eval'):
            model.eval()  # Set to evaluation mode
            st.success("✅ Model loaded and set to evaluation mode!")
        else:
            st.warning("⚠️ Model loaded but doesn't have eval() method")
        
        # Show model info if debug is enabled
        if debug:
            try:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                st.info(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
            except:
                st.info("📊 Could not calculate model parameters")
        
        return model, device
        
    except FileNotFoundError:
        st.error(f"❌ File not found: {model_path}")
        return None, device
    except RuntimeError as e:
        if "PytorchStreamReader" in str(e):
            st.error("❌ Corrupted model file. Please re-download the model.")
        else:
            st.error(f"❌ Runtime error loading model: {str(e)}")
        return None, device
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {str(e)}")
        st.info("💡 Try uploading a fresh copy of the model file")
        return None, device

# Usage examples:
if __name__ == "__main__":
    # Basic usage
    model, device = load_model()
    
    # With debugging enabled
    # model, device = load_model(debug=True)
    
    # With custom model name
    # model, device = load_model('my_custom_model.pt', debug=True)
    
    if model is not None:
        st.success("🎉 Model ready for inference!")
    else:
        st.error("🚫 Failed to load model. Check the logs above.")

    # File Upload Section
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
        image = Image.open(uploaded_image).convert('RGB')
        st.image(
            image,
            caption=f'{uploaded_image.name}',
            use_container_width=True
        )

        # Image info
        st.markdown(f"""
        <div class="info-box">
            <strong>Image Details:</strong><br>
            Size : {image.size[0]} x {image.size[1]} pixels<br>
            Format : {uploaded_image.type}<br>
            File size : {len(uploaded_image.getvalue()) / 1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #666;">
            <h3>No Image Selected</h3>
            <p>Please upload an image to see the preview</p>
        </div>
        """, unsafe_allow_html=True)

# Prediction Section
if uploaded_image is not None and model is not None:
    st.markdown("---")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button('Analyze Emotion', use_container_width=True)

    if predict_button:
        with st.spinner('Analyzing emotion... Please wait'):
            try:
                class_name = ['Fear', 'Happy', 'Neutral', 'Sad']
                emoji_map = {'Fear': '😨', 'Happy': '😊', 'Neutral': '😐', 'Sad': '😢'}
                color_map = {'Fear': 'emotion-fear', 'Happy': 'emotion-happy',
                             'Neutral': 'emotion-neutral', 'Sad': 'emotion-sad'}

                # Get prediction
                probli = pred_class(model, image, class_name)
                max_index = np.argmax(probli[0])

                # Results Section
                st.markdown("## Prediction Results")

                # Create two columns for results
                result_col1, result_col2 = st.columns([2, 1])

                with result_col1:
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

                    # Display results with styling
                    for i, (emotion, prob) in enumerate(zip(class_name, probli[0])):
                        emoji = emoji_map[emotion]
                        percentage = prob * 100
                        is_max = (i == max_index)

                        # Create styled result
                        if is_max:
                            st.markdown(f"""
                            <div class="emotion-result {color_map[emotion]}" style="border: 3px solid #gold;">
                                <h3 style="margin:0; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                                    {emoji} <strong>{emotion}</strong>: {percentage:.1f}%
                                </h3>
                                <p style="margin:0; color: white; font-size: 0.9em;">Primary Detection</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="padding: 0.5rem; margin: 0.3rem 0; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #ddd;">
                                <span style="font-size: 1.1em;">{emoji} {emotion}: <strong>{percentage:.1f}%</strong></span>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                with result_col2:
                    # Create a donut chart
                    fig = go.Figure(data=[go.Pie(
                        labels=[f"{emoji_map[emotion]} {emotion}" for emotion in class_name],
                        values=[prob * 100 for prob in probli[0]],
                        hole=.3,
                        marker_colors=['#ff6b6b', '#56ab2f', '#95a5a6', '#4a90e2']
                    )])

                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(
                        title="Emotion Distribution",
                        annotations=[dict(text='Confidence', x=0.5, y=0.5, font_size=16, showarrow=False)],
                        height=400,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Confidence indicator
                max_confidence = probli[0][max_index] * 100
                if max_confidence > 80:
                    confidence_color = "green"
                    confidence_text = "High Confidence"
                elif max_confidence > 60:
                    confidence_color = "orange"
                    confidence_text = "Medium Confidence"
                else:
                    confidence_color = "red"
                    confidence_text = "Low Confidence"

                st.markdown(f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <span style="background: {confidence_color}; color: white; padding: 0.5rem 1rem; 
                    border-radius: 20px; font-weight: bold;">
                        {confidence_text}: {max_confidence:.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please make sure the model file exists and the prediction function works correctly.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Powered by <strong>Deep Learning</strong> | Built with using <strong>Streamlit</strong></p>
    <p><small>For best results, use clear images with visible faces</small></p>
</div>

""", unsafe_allow_html=True)








