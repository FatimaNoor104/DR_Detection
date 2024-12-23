import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize

def generate_grad_cam(model, input_tensor, image_path):
    def save_gradients(gradients):
        global grad_value
        grad_value = gradients
    last_conv_layer = model.layer4[2].conv3

    def forward_hook(module, input, output):
        global activations
        activations = output
        output.register_hook(save_gradients)
    last_conv_layer.register_forward_hook(forward_hook)

    output = model(input_tensor)
    class_idx = torch.argmax(output, dim=1).item()
    model.zero_grad()
    output[0, class_idx].backward()

    weights = torch.mean(grad_value, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze(0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.detach().numpy()

    heatmap_resized = resize(cam, (224, 224), preserve_range=True)

    original_image = Image.open(image_path).convert("RGB")
    plt.imshow(original_image)
    plt.imshow(heatmap_resized, cmap="jet", alpha=0.5)
    plt.axis("off")

    grad_cam_path = "uploads/grad_cam.jpg"
    plt.savefig(grad_cam_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return grad_cam_path
