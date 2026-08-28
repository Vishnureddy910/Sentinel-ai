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
        
        # Normalize to 0-1 for visualization
        heatmap = heatmap / torch.max(heatmap)
        return heatmap.cpu().numpy()

if __name__ == "__main__":
    print("--- Phase 3: Generating Grad-CAM Heatmap ---")
    
    # Load the model and set to evaluation mode
    model = get_model()
    model.eval()
    
    # The roadmap requirement: hook layer4[1].conv2
    target_layer = model.layer4[1].conv2
    cam = GradCAM(model, target_layer)
    
    # Load a dummy image from Client 1
    img_path = r"data\client_1\dummy_0001.png"
    original_image = cv2.imread(img_path)
    
    # Add some gray to our black dummy image so we can actually see the heatmap overlay
    original_image = cv2.add(original_image, np.ones_like(original_image) * 50)
    
    # Preprocess the image
    rgb_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(pil_img).unsqueeze(0) 
    
    # Generate heatmap for class 0 (Atelectasis)
    print("Running forward and backward hooks...")
    heatmap = cam.generate_heatmap(input_tensor, target_class=0)
    
    # 5. Basic OpenCV overlay
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Blend original image and heatmap
    blended = cv2.addWeighted(original_image, 0.5, heatmap_colored, 0.5, 0)
    
    output_path = r"data\sample_test\gradcam_output.png"
    cv2.imwrite(output_path, blended)
    print(f"Success! Visual confirmed. Heatmap saved to: {output_path}")