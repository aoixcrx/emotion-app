from typing import List, Tuple
import torch
from torchvision import transforms
from PIL import Image

def pred_class(model: torch.nn.Module, image: Image.Image, class_names: List[str],
               image_size: Tuple[int, int] = (300, 300), device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. ส่งโมเดลไป device และบังคับให้เป็น float32
    model.to(device)
    model = model.float()  # บังคับให้ model เป็น float32
    model.eval()
    
    # Debug: ตรวจสอบ model dtype
    print(f"Model dtype after conversion: {next(model.parameters()).dtype}")
    
    # 2. สร้าง transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 3. แปลง image เป็น tensor + batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor = input_tensor.float()  # บังคับให้ input เป็น float32
    
    # Debug: ตรวจสอบ input dtype
    print(f"Input tensor dtype: {input_tensor.dtype}")
    
    # 4. inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
    
    # 5. ส่งคืน probabilities ทั้งหมดเป็น list
    return [probs[0].cpu().numpy()]
