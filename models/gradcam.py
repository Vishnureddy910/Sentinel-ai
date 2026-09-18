import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Import our ResNet baseline from Phase 2
from models.resnet_model import get_model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 1. Forward and backward hooks to "tap into" the layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class):
        # Forward pass
        model_output = self.model(input_tensor)
        
        self.model.zero_grad()
        
        # Target the specific disease class for the backward pass
        target = model_output[0][target_class]
        target.backward()
        
        # 2. Grad-CAM Math: Global average pool the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # 3. Multiply gradients (importance weights) into the feature maps
        activations = self.activations.detach()[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # 4. Sum channels and apply ReLU (only positive influence matters)
        heatmap = torch.sum(activations, dim=0)
        heatmap = F.relu(heatmap)
        
        # Normalize to 0-1 for visualization. Guard the divide: if no activation
        # has positive influence the ReLU leaves an all-zero map.
        peak = torch.max(heatmap)
        if peak > 0:
            heatmap = heatmap / peak
        return heatmap.detach().cpu().numpy()

if __name__ == "__main__":
    import sys
    from pathlib import Path

    from models.dataset import DISEASES, inference_transform

    ROOT = Path(__file__).resolve().parent.parent
    print("--- Generating Grad-CAM Heatmap ---")

    # Load the federated global model rather than an untrained backbone
    model = get_model(pretrained=False)
    for ckpt in [ROOT / "data" / "models" / "global_best.pth",
                 ROOT / "data" / "models" / "global_latest.pth"]:
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"Loaded weights: {ckpt.name}")
            break
    else:
        print("WARNING: no trained checkpoint found, heatmap will be meaningless")
    model.eval()

    target_layer = model.layer4[1].conv2
    cam = GradCAM(model, target_layer)

    # Use a real prepared X-ray (first val image) unless one is passed in
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        import pandas as pd
        client_dir = ROOT / "data" / "client_1_prep"
        img_path = client_dir / pd.read_csv(client_dir / "val.csv").iloc[0]["Image Index"]
    print(f"Image: {img_path}")

    pil_img = Image.open(img_path).convert("RGB")
    input_tensor = inference_transform()(pil_img).unsqueeze(0)

    # Explain whichever disease the model considers most likely
    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor))[0]
    target_class = int(torch.argmax(probs))
    print(f"Explaining top prediction: {DISEASES[target_class]} ({probs[target_class]:.3f})")

    heatmap = cam.generate_heatmap(input_tensor, target_class=target_class)

    # Overlay on the 224x224 view the model actually saw
    base = np.array(pil_img.resize((256, 256)))[16:240, 16:240]
    base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(base, 0.6, heatmap_colored, 0.4, 0)

    out_dir = ROOT / "data" / "sample_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "gradcam_output.png"
    cv2.imwrite(str(output_path), blended)
    print(f"Saved heatmap to: {output_path}")
