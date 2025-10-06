# Dice Detector Documentation

This directory contains detailed documentation for the dice detector project.

## Contents

- [PRD.md](PRD.md) - Product Requirements Document with complete project specification
- [Main README](../README.md) - Quick start guide and usage instructions

## Quick Links

### Getting Started

1. [Installation](../README.md#installation)
2. [Quick Start](../README.md#quick-start)
3. [Environment Check](../README.md#setup)

### Workflow

The typical workflow follows these steps:

```
1. Prepare Data → 2. Build Dataset → 3. Train Model → 4. Evaluate
     ↓                    ↓                  ↓              ↓
 manifest.yaml      coco.yaml        best.pt       metrics.json
```

Optionally, insert distillation between steps 2 and 3:

```
2.5. Distillation (DINOv3 → YOLO Backbone)
          ↓
  student_backbone.pt
```

### Scripts Reference

#### build_dataset.py

**Purpose**: Merge multiple dataset roots and create train/val/test splits

**Key Arguments**:
- `--manifest`: Path to dataset manifest YAML (required)
- `--output-dir`: Output directory (default: `data/processed/<run_id>`)
- `--export-format`: Export format (coco, ultralytics, or both)
- `--max-images`: Limit dataset size for debugging
- `--no-deduplicate`: Disable image deduplication

**Output**:
- `data/processed/<run_id>/`
  - `coco.yaml` - Ultralytics data config
  - `coco.{train,val,test}.json` - COCO format annotations
  - `images/{train,val,test}/` - Organized images
  - `splits.yaml` - Split assignments
  - `summary.json` - Dataset statistics

#### train.py

**Purpose**: Train YOLO detector on labeled data

**Key Arguments**:
- `--data`: Dataset YAML file (required)
- `--model`: Model name or checkpoint
- `--backbone-weights`: Distilled backbone weights (optional)
- `--epochs`: Training epochs
- `--batch`: Batch size
- `--imgsz`: Image size

**Output**:
- `runs/detect/<name>/`
  - `weights/best.pt` - Best model weights
  - `weights/last.pt` - Last checkpoint
  - `results.png` - Training curves
  - `confusion_matrix.png` - Confusion matrix

#### test.py

**Purpose**: Evaluate trained model

**Key Arguments**:
- `--data`: Dataset YAML file (required)
- `--weights`: Model weights (required)
- `--plots`: Generate visualization plots
- `--save-json`: Save COCO format results
- `--profile`: Profile inference speed

**Output**:
- `runs/detect/<name>/`
  - `metrics.json` - Evaluation metrics
  - `PR_curve.png` - Precision-Recall curve
  - `confusion_matrix.png` - Confusion matrix

#### distill.py

**Purpose**: Framework for DINOv3 → YOLO distillation

**Key Arguments**:
- `--manifest` or `--unlabeled-dirs`: Unlabeled data source
- `--teacher`: DINOv3 variant (dinov3_s/b/l/g)
- `--student`: YOLO variant
- `--epochs`: Distillation epochs
- `--feature-layers`: Layers to align

**Output**:
- `runs/distill/<run_id>/`
  - `student_backbone.pt` - Distilled backbone (placeholder)
  - `config.yaml` - Distillation configuration

#### env_check.py

**Purpose**: Validate environment setup

**Checks**:
- Python version (3.10+)
- PyTorch installation
- CUDA availability
- Required dependencies
- Project structure

## Manifest File Format

The manifest YAML defines datasets, splits, and export settings:

```yaml
# Unique identifier for this experiment
run_id: my_experiment_001

# Class names (single class for dice detection)
classes: [dice]

# Label format (currently supports datamuro)
label_format: datamuro

# Dataset roots
roots:
  # Labeled datasets (with annotations)
  labeled:
    - path: E:/data/dice/labeled_set1
    - path: D:/datasets/dice_labeled_set2
    # Add more labeled roots as needed
  
  # Unlabeled datasets (for distillation)
  unlabeled:
    - path: E:/data/dice/unlabeled
    - path: \\nas\share\dice_raw
    # Add more unlabeled roots as needed

# Split configuration
splits:
  train_ratio: 0.8  # 80% for training
  val_ratio: 0.1    # 10% for validation
  test_ratio: 0.1   # 10% for testing

# Random seed for reproducibility
seed: 42

# Export format
export:
  format: coco  # Options: coco, ultralytics, both
```

## Configuration Files

Configuration files in `configs/` provide default settings:

- `dataset.yaml` - Dataset processing settings
- `train.yaml` - Training hyperparameters
- `test.yaml` - Evaluation settings
- `distill.yaml` - Distillation configuration

These can be overridden via command-line arguments.

## Data Formats

### DataMuro Format

Flexible annotation format supporting:

1. **COCO-style** (single JSON):
   - `images` array with image metadata
   - `annotations` array with bboxes and classes

2. **Per-image JSON**:
   - One `.json` file per image
   - Contains annotations for that image

3. **Frame-based** (video):
   - `frames` array with per-frame annotations

The adapter automatically detects and parses these formats.

### COCO Export Format

Standard COCO JSON format with:
- Image metadata (width, height, file path)
- Annotations (bbox in [x, y, width, height], category_id)
- Categories (class definitions)

### Ultralytics Format

YOLO native format:
- Images in `images/{train,val,test}/`
- Labels in `labels/{train,val,test}/` as `.txt` files
- Format: `class_id center_x center_y width height` (normalized)

## Metrics Explanation

### mAP (mean Average Precision)

- **mAP@.5**: IoU threshold of 0.5 (easier threshold)
- **mAP@[.5:.95]**: Average across IoU thresholds 0.5 to 0.95 (stricter, COCO standard)

### Precision and Recall

- **Precision**: Of all detections, how many were correct?
  - High precision = few false positives
- **Recall**: Of all ground truth objects, how many were detected?
  - High recall = few false negatives

### Trade-off

- Lowering confidence threshold increases recall but decreases precision
- PR curve shows this trade-off at different thresholds

## Best Practices

### Dataset Preparation

1. **Organize by source**: Keep different data sources in separate roots
2. **Clean annotations**: Verify bbox coordinates and class labels
3. **Balance classes**: Ensure roughly equal representation (for multi-class)
4. **Diverse data**: Include various lighting, angles, backgrounds

### Training

1. **Baseline first**: Always establish baseline before trying distillation
2. **Small to large**: Start with small model (n), scale up if needed
3. **Monitor overfitting**: Watch val loss vs train loss
4. **Augmentation**: Balance between regularization and data quality
5. **Learning rate**: Start with default, adjust if convergence issues

### Evaluation

1. **Test set only**: Never train on test data
2. **Multiple metrics**: Don't rely on single metric
3. **Visual inspection**: Look at predictions, not just numbers
4. **Edge cases**: Test on difficult scenarios
5. **Real-world validation**: Test in actual deployment conditions

## Troubleshooting

### Dataset Issues

**Problem**: No annotations found  
**Solution**: 
- Check annotation format matches DataMuro expectations
- Verify `.json` files exist
- Use `--verbose` for detailed logging

**Problem**: Images not found  
**Solution**:
- Check paths in manifest are correct
- Verify image extensions are supported
- Ensure paths are accessible (network shares mounted)

### Training Issues

**Problem**: CUDA out of memory  
**Solution**:
- Reduce `--batch` size
- Reduce `--imgsz`
- Use smaller model variant

**Problem**: Poor convergence  
**Solution**:
- Check learning rate (try 0.001 or 0.0001)
- Verify labels are correct
- Increase epochs
- Check for class imbalance

**Problem**: Overfitting  
**Solution**:
- More data augmentation
- Smaller model
- Early stopping (patience)
- More training data

### Evaluation Issues

**Problem**: Low mAP but visually good predictions  
**Solution**:
- Check IoU threshold (might be too strict)
- Verify ground truth annotations are accurate
- Consider if metric matches your use case

**Problem**: High precision, low recall  
**Solution**:
- Lower confidence threshold
- Check if small objects are being missed
- Verify all ground truth objects are annotated

## Extending the Project

### Adding New Data Formats

1. Implement new adapter in `src/data/`
2. Follow `DataMuroAdapter` pattern
3. Parse to internal schema (see `schemas.py`)
4. Register in `build_dataset.py`

### Adding New Export Formats

1. Implement new exporter in `src/data/`
2. Follow `COCOExporter` pattern
3. Add to `build_dataset.py` options

### Implementing Full Distillation

The distillation script is a framework. To implement:

1. Load DINOv3 model and extract features
2. Load YOLO backbone and extract features
3. Implement projection heads for alignment
4. Implement distillation loss (cosine/L2)
5. Training loop with backpropagation
6. Save student backbone weights

Consider using Lightly Train platform for managed distillation.

## Additional Resources

- [PRD.md](PRD.md) - Detailed requirements and specifications
- [Ultralytics Docs](https://docs.ultralytics.com/)
- [COCO Format](https://cocodataset.org/#format-data)
- [DINOv3 Paper](https://arxiv.org/abs/2304.07193)

## Support

For issues or questions:
1. Check this documentation
2. Review error messages with `--verbose`
3. Run `python scripts/env_check.py`
4. Check GitHub issues
5. Open new issue with details

