import torch
import torch.nn as nn
import torchvision.models as models
import json
import os

class FromSoftClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(FromSoftClassifier, self).__init__()
        # Load pretrained ResNet50
        print("Loading ResNet50 weights...")
        self.model = models.resnet50(weights="IMAGENET1K_V1")
        
        # Freeze base layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def setup_model_and_loss(device_name="cpu"):
    # Load class weights
    weights_path = 'class_weights.json'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"{weights_path} not found. Run prepare_dataset.py first.")
        
    with open(weights_path, 'r') as f:
        class_weights_dict = json.load(f)
        
    # Sort weights by key to ensure correct order
    sorted_classes = sorted(class_weights_dict.keys())
    weights_list = [class_weights_dict[cls] for cls in sorted_classes]
    
    device = torch.device(device_name)
    weights_tensor = torch.tensor(weights_list, dtype=torch.float).to(device)
    
    print(f"Setting up model for {len(sorted_classes)} classes on {device}...")
    model = FromSoftClassifier(num_classes=len(sorted_classes))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    return model, criterion

if __name__ == "__main__":
    # verification
    print("Verifying model architecture...")
    model = FromSoftClassifier()
    # Dummy tensor with shape (1, 3, 360, 640) as per user's image dimensions
    dummy_input = torch.randn(1, 3, 360, 640)
    print(f"Passing dummy input of shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    if output.shape == (1, 7):
        print("Verification SUCCESS: Output shape is correct.")
    else:
        print("Verification FAILED: Output shape mismatch.")
