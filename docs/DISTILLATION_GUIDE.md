# Knowledge Distillation Implementation Guide

## Overview

Distilling DINOv3 teacher into YOLO student backbone requires:
1. Loading DINOv3 (teacher) - frozen weights
2. Loading YOLO backbone (student) - trainable
3. Feature alignment layers
4. Distillation loss
5. Training loop on unlabeled data

## Implementation Checklist

### 1. Load DINOv3 Teacher

```python
import torch

# Option A: Use official DINOv3
# Clone: git clone https://github.com/facebookresearch/dinov3
# Or install: pip install git+https://github.com/facebookresearch/dinov3

# Load pretrained DINOv3
teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
teacher.eval()
teacher.requires_grad_(False)

# Move to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher = teacher.to(device)
```

### 2. Load YOLO Student Backbone

```python
from ultralytics import YOLO

# Load YOLO model
student_model = YOLO('yolov8n.pt')

# Access backbone (this is tricky - YOLO wraps the model)
student_backbone = student_model.model.model[:10]  # First 10 layers = backbone
student_backbone.train()
student_backbone.requires_grad_(True)
```

### 3. Feature Alignment (Projection Heads)

```python
import torch.nn as nn

class ProjectionHead(nn.Module):
    """Project features from different dimensions to same space."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x):
        return self.proj(x)

# Create projection heads for each layer pair
teacher_dim = 768  # DINOv3-B output dimension
student_dim = 256  # YOLO backbone dimension (varies by layer)

proj_heads = nn.ModuleList([
    ProjectionHead(student_dim, teacher_dim) for _ in range(3)
])
proj_heads = proj_heads.to(device)
```

### 4. Distillation Loss

```python
import torch.nn.functional as F

def distillation_loss(student_features, teacher_features, temperature=4.0):
    """
    Compute distillation loss between student and teacher features.
    
    Args:
        student_features: List of student feature maps
        teacher_features: List of teacher feature maps
        temperature: Temperature for softening distributions
    
    Returns:
        Loss value
    """
    total_loss = 0.0
    
    for s_feat, t_feat in zip(student_features, teacher_features):
        # Normalize features
        s_feat = F.normalize(s_feat, dim=1)
        t_feat = F.normalize(t_feat, dim=1)
        
        # Cosine similarity loss
        loss = 1 - F.cosine_similarity(s_feat, t_feat, dim=1).mean()
        total_loss += loss
    
    return total_loss / len(student_features)
```

### 5. Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Data loading
transform = transforms.Compose([
    transforms.Resize(640),
    transforms.CenterCrop(640),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Load unlabeled images
# These are from your manifest's unlabeled roots
unlabeled_image_paths = [...]  # Load from unlabeled_images.yaml

dataset = UnlabeledDataset(unlabeled_image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Optimizer
optimizer = optim.AdamW(
    list(student_backbone.parameters()) + list(proj_heads.parameters()),
    lr=1e-4,
    weight_decay=0.01
)

# Training loop
num_epochs = 50

for epoch in range(num_epochs):
    student_backbone.train()
    proj_heads.train()
    
    total_loss = 0.0
    
    for batch_idx, images in enumerate(dataloader):
        images = images.to(device)
        
        # Forward pass through teacher (no gradients)
        with torch.no_grad():
            teacher_features = teacher.forward_features(images)
        
        # Forward pass through student
        student_features = []
        x = images
        for layer in student_backbone:
            x = layer(x)
            student_features.append(x)
        
        # Project student features
        projected_features = [
            proj(feat) for proj, feat in zip(proj_heads, student_features[-3:])
        ]
        
        # Compute loss
        loss = distillation_loss(projected_features, [teacher_features] * 3)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'student_backbone': student_backbone.state_dict(),
            'proj_heads': proj_heads.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f'runs/distill/checkpoint_epoch_{epoch+1}.pt')

# Save final distilled backbone
torch.save(student_backbone.state_dict(), 'runs/distill/student_backbone.pt')
```

### 6. Use Distilled Backbone in YOLO Training

```python
# This requires modifying YOLO's model initialization
# Currently not fully supported in the framework

# Theoretical usage:
# python scripts/train.py \
#     --data data/processed/dice_exp_001/coco.yaml \
#     --model yolov8n.pt \
#     --backbone-weights runs/distill/student_backbone.pt \
#     --epochs 100
```

## Challenges

1. **Feature Dimension Mismatch**
   - DINOv3: ViT architecture with token-based features
   - YOLO: CNN architecture with spatial feature maps
   - Need careful alignment and projection

2. **Layer Mapping**
   - Which teacher layers align with which student layers?
   - Requires experimentation

3. **YOLO Model Access**
   - Ultralytics wraps models, making backbone access non-trivial
   - May need to modify YOLO source code

## Alternatives

### Use Lightly Train Platform

Lightly provides managed distillation:
- Web-based interface
- Handles feature alignment automatically
- Exports compatible backbones

### Use Pre-trained Backbones

Modern YOLO models already use strong ImageNet pretrained backbones.
The benefit of distillation may be marginal unless:
- You have LOTS of unlabeled data (>100K images)
- Your domain is very different from ImageNet
- You're optimizing for edge deployment (smaller models)

## Recommended Approach

For your dice detection project:

1. **Start with baseline** (no distillation)
   ```bash
   python scripts/train.py --data data/processed/dice_exp_001/coco.yaml --model yolov8n.pt --epochs 100
   ```

2. **Evaluate performance**
   ```bash
   python scripts/test.py --data data/processed/dice_exp_001/coco.yaml --weights runs/detect/train/weights/best.pt --plots
   ```

3. **Only if baseline isn't sufficient**, implement distillation or use Lightly

The baseline should give good results with:
- 15,849 labeled images
- 71 classes
- Modern YOLO architecture

Distillation is more valuable when:
- Labeled data is scarce (< 1000 images)
- Unlabeled data is abundant (> 50K images)
- Need to compress model for deployment

