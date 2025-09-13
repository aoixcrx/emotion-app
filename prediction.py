from typing import List, Tuple
import torch
from torchvision import transforms
from PIL import Image

def pred_class(model: torch.nn.Module, image: Image.Image, class_names: List[str],
               image_size: Tuple[int, int] = (300, 300), device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. ส่งโมเดลไป device
    model.to(device)
    model.eval()
    
    # 2. สร้าง transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 3. แปลง image เป็น tensor + batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 4. แปลง input ให้ตรงกับ model precision
    try:
        model_dtype = next(model.parameters()).dtype
        if model_dtype == torch.float16:
            input_tensor = input_tensor.half()
        elif model_dtype == torch.float32:
            input_tensor = input_tensor.float()
    except Exception as e:
        # ถ้าไม่สามารถตรวจสอบ dtype ได้ ให้ใช้ float32
        input_tensor = input_tensor.float()
    
    # 5. inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
    
    # 5. ส่งคืน probabilities ทั้งหมดเป็น list
    return [probs[0].cpu().numpy()]
