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
    
    # 4. inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
    
    # 5. ดึง class และ probability
    pred_label_idx = torch.argmax(probs, dim=1).item()   # convert tensor -> int
    pred_classname = class_names[pred_label_idx]
    pred_prob = probs[0, pred_label_idx].item()          # probability ของ class ที่ predict

    return pred_classname, pred_prob
