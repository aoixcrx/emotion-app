from typing import List, Tuple
import torch
from torchvision import transforms
from PIL import Image
import warnings

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
    
    # 4. จัดการ dtype mismatch ด้วยวิธีที่ปลอดภัย
    # ปิด warning เกี่ยวกับ dtype
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # บังคับให้ model และ input ใช้ float32
    model = model.float()
    input_tensor = input_tensor.float()
    
    # 4. inference
    with torch.no_grad():
        try:
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
        except RuntimeError as e:
            if "expected scalar type Half but found Float" in str(e):
                # ถ้า error เกี่ยวกับ Half/Float ให้ลองใช้ double
                model = model.double()
                input_tensor = input_tensor.double()
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
            else:
                raise e
    
    # 5. ส่งคืน probabilities ทั้งหมดเป็น list
    return [probs[0].cpu().numpy()]
