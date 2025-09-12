# prediction.py
from typing import List, Tuple
import torch
import torchvision.transforms as T
from PIL import Image


def pred_class(
    model: torch.nn.Module,
    image: Image.Image,
    class_names: List[str],
    image_size: Tuple[int, int] = (224, 224),
):
    """
    Predict class & probability from an input image
    Args:
        model: Trained PyTorch model
        image: PIL Image
        class_names: List of emotion class names
        image_size: Target resize for the model
    Returns:
        probs: list of probabilities
        classname: predicted class name
    """

    # 🔹 Transformation
    image_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # 🔹 Ensure model on correct device
    # device = next(model.parameters()).device
    # model = model.to(torch.float32).eval()   # force float32 + eval mode
    device = next(model.parameters()).device
    model = model.to(device).float().eval()   # force float32


    with torch.inference_mode():
        # Transform image → [1, C, H, W]
        transformed_image = image_transform(image).unsqueeze(0).to(device).to(torch.float32)

        # Forward pass
        outputs = model(transformed_image)

        # Logits → Probabilities
        probs_tensor = torch.softmax(outputs, dim=1)

        # Predicted label index
        pred_idx = torch.argmax(probs_tensor, dim=1).item()

        classname = class_names[pred_idx]
        probs = probs_tensor.cpu().numpy().tolist()[0]

    return probs, classname

# ## Making Pridcition return class & prob
# from typing import List, Tuple
# import torch
# import torchvision.transforms as T

# from PIL import Image
# def pred_class(model: torch.nn.Module,
#                         image,
#                         class_names: List[str],
#                         image_size: Tuple[int, int] = (224, 224),
#                         ):
    
    
#     # 2. Open image
#     img = image

#     # 3. Create transformation for image (if one doesn't exist)
#     image_transform = T.Compose([
#             T.Resize(image_size),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]),
#         ])
    

#     ### Predict on image ### 
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 4. Make sure the model is on the target device
#     model.to(device)
    

#     # 5. Turn on model evaluation mode and inference mode
#     model.eval()
#     with torch.inference_mode():
#       # # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
#       # transformed_image = image_transform(img).unsqueeze(dim=0).float()

#       # # 7. Make a prediction on image with an extra dimension and send it to the target device
#       # target_image_pred = model(transformed_image.to(device))
#       transformed_image = image_transform(img).unsqueeze(0).to(device).to(torch.float32)
#       target_image_pred = model(transformed_image)


#       # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
#       target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

#       # 9. Convert prediction probabilities -> prediction labels
#       target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

#       classname =  class_names[target_image_pred_label]
#       prob = target_image_pred_probs.cpu().numpy()

#     return prob







