import torch
import torch.nn as nn
from torchvision import models
from utils.grad_cam import generate_grad_cam

def load_model():
    class_names = {0: "DR", 1: "No_DR"}
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )
    try:
        state_dict = torch.load("model/model.pth", map_location=torch.device("cpu"))
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    model.eval()
    return model, class_names
    
def predict_and_visualize(model, input_tensor, image_path, class_names):
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()
        predicted_class_idx = outputs.argmax(dim=1).item()
        predicted_label = class_names[predicted_class_idx]

    grad_cam_image = generate_grad_cam(model, input_tensor, image_path)
    return predicted_label, probabilities, grad_cam_image
