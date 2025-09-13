## Making Pridcition return class & prob
from typing import List, Tuple
import torch
import torchvision.transforms as T

from PIL import Image


def pred_class(model: torch.nn.Module,
               image,
               class_names: List[str],
               image_size: Tuple[int, int] = (224, 224),
               ):
    # 2. Open image (image เป็น PIL.Image)
    # 3. Create transformation for image
    image_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.inference_mode():
        # 4. Transform image and add batch dimension
        transformed_image = image_transform(image).unsqueeze(dim=0)
        # 5. แปลง dtype ให้ตรงกับโมเดล
        transformed_image = transformed_image.to(next(model.parameters()).dtype)
        output = model(transformed_image.to(device))
        target_image_pred_probs = torch.softmax(output, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    prob = target_image_pred_probs.cpu().numpy()
    return prob