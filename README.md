# Dice Object Detector with DINOv3 Pretraining and YOLO Fine-Tuning

End-to-end training and testing workflow for detecting dice in images using knowledge distillation from DINOv3 to YOLOv12.

## Features

- **Multi-Root Dataset Merging**: Combine labeled and unlabeled datasets from multiple directories
- **DataMuro Format Support**: Flexible annotation format adapter with COCO export
- **Deterministic Splitting**: Reproducible train/val/test splits with deduplication
- **Knowledge Distillation**: DINOv3 → YOLOv12 backbone distillation (framework provided)
- **Baseline Training**: Standard YOLO fine-tuning on labeled data
- **Comprehensive Evaluation**: mAP metrics, plots, and performance profiling

## Project Structure

```
dinov3_exp/
├── configs/                    # Configuration files
│   ├── dataset.yaml           # Dataset processing config
│   ├── distill.yaml           # Distillation config
│   ├── train.yaml             # Training config
│   └── test.yaml              # Testing config
├── manifests/                  # Dataset manifests
│   └── dice.yaml              # Example dice dataset manifest
├── src/                        # Source code
│   ├── data/                  # Data processing modules
│   │   ├── datamuro_adapter.py
│   │   ├── dataset_merger.py
│   │   ├── dataset_splitter.py
│   │   ├── coco_exporter.py
│   │   └── schemas.py
│   └── utils/                 # Utility functions
│       ├── path_utils.py
│       ├── hash_utils.py
│       ├── yaml_utils.py
│       └── logger.py
├── scripts/                    # Executable scripts
│   ├── build_dataset.py       # Build unified dataset
│   ├── train.py               # Train YOLO detector
│   ├── test.py                # Evaluate model
│   ├── distill.py             # Knowledge distillation
│   └── env_check.py           # Environment validation
├── docs/                       # Documentation
│   └── PRD.md                 # Product Requirements Document
├── data/                       # Data directory (created automatically)
│   └── processed/             # Processed datasets
└── runs/                       # Training/testing runs (created automatically)
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support (recommended)
- Windows 10+ (designed for Windows, but should work on Linux/macOS)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd dinov3_exp
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

For CUDA support, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. **Verify installation**:
```bash
python scripts/env_check.py
```

## Quick Start

### 1. Prepare Dataset Manifest

Create a manifest file (e.g., `manifests/my_dice.yaml`) describing your datasets:

```yaml
run_id: dice_exp_001
classes: [dice]
label_format: datamuro

roots:
  labeled:
    - path: E:/data/dice/labeled_set1
    - path: D:/datasets/dice_labeled_set2
  unlabeled:
    - path: E:/data/dice/unlabeled

splits:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

seed: 42

export:
  format: coco
```

### 2. Build Dataset

Merge all dataset roots and create train/val/test splits:

```bash
python scripts/build_dataset.py --manifest manifests/my_dice.yaml
```

This will:
- Merge multiple dataset roots
- Deduplicate images by content hash
- Split data deterministically
- Export to COCO format
- Save to `data/processed/dice_exp_001/`

### 3. Train Baseline Model

Train a YOLO detector on labeled data:

```bash
python scripts/train.py \
    --data data/processed/dice_exp_001/coco.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
```

### 4. Evaluate Model

Test the trained model:

```bash
python scripts/test.py \
    --data data/processed/dice_exp_001/coco.yaml \
    --weights runs/detect/train/weights/best.pt \
    --plots
```

## Advanced Usage

### Knowledge Distillation (DINOv3 → YOLO)

**Note**: Full distillation implementation requires additional setup. The script provides a framework and configuration.

```bash
python scripts/distill.py \
    --manifest manifests/my_dice.yaml \
    --teacher dinov3_b \
    --student yolov12n \
    --epochs 50 \
    --batch-size 32 \
    --imgsz 640
```

Then train YOLO with distilled backbone:

```bash
python scripts/train.py \
    --data data/processed/dice_exp_001/coco.yaml \
    --model yolov12n.pt \
    --backbone-weights runs/distill/dice_exp_001/student_backbone.pt \
    --epochs 100
```

### Using Configuration Files

You can use YAML configuration files for reproducible experiments:

```bash
python scripts/train.py \
    --data data/processed/dice_exp_001/coco.yaml \
    --config configs/train.yaml
```

### Performance Profiling

Profile inference speed and throughput:

```bash
python scripts/test.py \
    --data data/processed/dice_exp_001/coco.yaml \
    --weights runs/detect/train/weights/best.pt \
    --profile \
    --profile-iterations 100
```

### Dataset Options

**Disable deduplication**:
```bash
python scripts/build_dataset.py \
    --manifest manifests/my_dice.yaml \
    --no-deduplicate
```

**Limit dataset size (for debugging)**:
```bash
python scripts/build_dataset.py \
    --manifest manifests/my_dice.yaml \
    --max-images 100
```

**Export to multiple formats**:
```bash
python scripts/build_dataset.py \
    --manifest manifests/my_dice.yaml \
    --export-format both
```

## Data Format: DataMuro

The DataMuro adapter supports various annotation formats. Your annotations can be in:

1. **COCO-like format** (single JSON file):
```json
{
  "images": [
    {"id": 1, "file_name": "image001.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [100, 150, 200, 250],
      "area": 50000
    }
  ]
}
```

2. **Per-image JSON files**: `image001.json` alongside `image001.jpg`

3. **Custom formats**: The adapter can be extended to support additional formats

## Experiment Tracking

All experiments are logged to the `runs/` directory:

```
runs/
├── detect/
│   ├── train/                 # Training run
│   │   ├── weights/
│   │   │   ├── best.pt       # Best model
│   │   │   └── last.pt       # Last checkpoint
│   │   ├── results.png       # Training curves
│   │   └── ...
│   └── test/                  # Testing run
│       ├── confusion_matrix.png
│       ├── PR_curve.png
│       └── metrics.json
└── distill/
    └── dice_exp_001/
        ├── student_backbone.pt
        └── config.yaml
```

## Metrics

The evaluation provides:

- **mAP@[.5:.95]**: Primary metric (COCO-style)
- **mAP@.5**: IoU threshold 0.5
- **Precision**: Correct detections / All detections
- **Recall**: Correct detections / All ground truth
- **Per-class metrics**: Individual class performance
- **Confusion matrix**: Visualization of classification errors
- **PR curves**: Precision-Recall trade-offs
- **Inference metrics**: Latency (ms), Throughput (FPS)

## Tips and Best Practices

### Dataset Preparation

1. **Use multiple dataset roots**: Organize data by source for easier management
2. **Leverage unlabeled data**: More unlabeled images → better distillation
3. **Validate splits**: Check class distribution across train/val/test
4. **Deduplicate**: Remove duplicate images to prevent data leakage

### Training

1. **Start with baseline**: Establish baseline performance before distillation
2. **Use appropriate model size**: Start small (yolov8n) for faster iteration
3. **Monitor validation metrics**: Watch for overfitting
4. **Adjust augmentation**: Balance between regularization and preserving features
5. **Early stopping**: Use patience to avoid overtraining

### Distillation

1. **More unlabeled data is better**: Target 5-10x labeled data amount
2. **Match image domains**: Unlabeled data should match target distribution
3. **Experiment with layer alignment**: Different layers capture different features
4. **Consider compute budget**: Distillation is computationally expensive

## Troubleshooting

### Common Issues

**CUDA out of memory**:
- Reduce batch size: `--batch 8`
- Reduce image size: `--imgsz 512`
- Use gradient accumulation

**Path not found errors on Windows**:
- Use forward slashes or escaped backslashes in YAML
- Avoid spaces in paths
- For UNC paths, use double backslashes: `\\\\server\\share`

**No annotations found**:
- Check DataMuro format matches expected structure
- Verify annotation files exist alongside images
- Use `--verbose` for detailed logging

**Poor model performance**:
- Check class distribution (imbalanced?)
- Validate annotations (correct format?)
- Increase training epochs
- Try different augmentation settings
- Verify dataset isn't too small

## Windows-Specific Notes

- **Long path support**: Enable in Windows if paths exceed 260 characters
- **UNC paths**: Supported for network shares (`\\\\server\\share`)
- **PowerShell**: Use backticks for line continuation instead of backslashes
- **CUDA**: Ensure NVIDIA drivers are up to date

## Contributing

This is an experimental project. To extend functionality:

1. Add new adapters in `src/data/`
2. Add new exporters for different formats
3. Implement full distillation training loop
4. Add additional augmentation strategies
5. Integrate with experiment tracking platforms (W&B, MLflow)

## License

[Specify your license]

## Acknowledgments

- **Ultralytics YOLO**: Object detection framework
- **DINOv3**: Self-supervised learning
- **Lightly**: SSL and distillation tools

## References

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [DINOv3 Paper](https://arxiv.org/abs/2304.07193)
- [YOLO Documentation](https://github.com/ultralytics/ultralytics)

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].

