## PRD: Dice Object Detector with DINOv3 Pretraining and Ultralytics YOLO Fine‑Tuning

### Overview
Build an end‑to‑end training and testing workflow to detect dice in images. The canonical strategy is to use Lightly Train to distill features from a DINOv3 teacher into a YOLOv12 student backbone using unlabeled data, then fine‑tune the YOLO detector on labeled data. Labeled data are provided in the DataMuro format and may be spread across multiple dataset roots; unlabeled data should also be supported. Fallbacks include baseline fine‑tuning and SSL curation if distillation is not feasible with the available tooling. The deliverable includes runnable scripts, clear CLIs, and documentation.

### Goals
- Fine‑tune a YOLO detector to localize dice (single class: `dice`).
- Canonical: Distill DINOv3 → YOLOv12 backbone via Lightly Train on unlabeled data, then fine‑tune detection on labeled data.
- Optionally use SSL curation/pseudo‑labels as a fallback or ablation.
- Support ingesting multiple datasets across different paths; merge and split deterministically.
- Provide reproducible training and testing scripts with consistent CLIs and config files.
- Track core detection metrics (mAP@[.5:.95], precision/recall), plus latency and throughput.

### Non‑Goals
- Instance segmentation, pose estimation, or dice face (pip count) recognition.
- Advanced active learning UI; only CLI‑driven sampling/curation.
- Large‑scale distributed training beyond single/multi‑GPU on one machine.

### Stakeholders / Users
- ML engineer experimenting with SSL pretraining and YOLO fine‑tuning.
- Researcher evaluating effect of unlabeled data on detection performance.

## Data

### Sources and Layout
- Labeled datasets: multiple roots containing images and DataMuro annotations.
- Unlabeled datasets: multiple roots containing only images.
- All roots provided via a manifest file; the system merges them into a unified, versioned working directory under `data/processed/<run_id>/`.

### Data Formats
- Labeled: DataMuro format (assumed bounding boxes + class labels). We will implement a `DataMuroAdapter` to parse into an internal schema. If needed, we can export to COCO for Ultralytics compatibility.
- Unlabeled: images only. Used for SSL pretraining and/or data curation.

### Data Manifest (YAML)
Users provide a manifest listing datasets and splits across arbitrary paths. Example:

```yaml
run_id: dice_exp_001
classes: [dice]
label_format: datamuro
roots:
  labeled:
    - path: E:/data/dice/set1
    - path: D:/datasets/dice_project/labeled_part_b
  unlabeled:
    - path: E:/data/dice/unlabeled_dump
    - path: \\nas\share\dice_raw
splits:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
seed: 42
export:
  format: coco  # coco|ultralytics|datamuro (internal) 
```

### Class Schema
- Single class: `dice` (ID=0). If future classes are added, the adapter and config will support extension.

### Merging and Splitting Policy
- Merge all labeled roots and de‑duplicate by image hash and relative path.
- Deterministic split by stable hashing seeded by `seed` in the manifest.
- Keep split leakage prevention: images with identical hashes are assigned to the same split.

### Augmentations
- Training: geometric flips/rotations, color jitter, mixup/mosaic as supported by Ultralytics, with configurable intensity.
- Validation/Test: resize/letterbox only, no label‑changing transforms.

## Approach

### Baseline
- Fine‑tune the latest Ultralytics YOLO detector on labeled data only to establish a baseline (e.g., `yolo train ...`).

### Knowledge Distillation (Lightly Train)
Use Lightly Train to distill features from a DINOv3 teacher into a YOLOv12 student backbone with unlabeled data:
- Teacher: DINOv3 (small/base) with fixed weights (public or briefly refined on unlabeled).
- Student: YOLOv12 backbone initialized from Ultralytics default or scratch.
- Distillation objective: align intermediate feature maps via projection heads with a mix of cosine/L2 losses; optionally include token/pooling level alignment. Train only the student during distillation.
- Data: unlabeled images from all specified roots; standard augmentations consistent with YOLO pretraining.
- Output: distilled YOLOv12 backbone weights (`student_backbone.pt`) compatible with Ultralytics detection head initialization.

Fallbacks and Ablations:
- If Lightly Train is unavailable, run local distillation using Lightly OSS utilities or skip to baseline.
- Optional SSL curation or pseudo‑labeling can be used as separate experiments.

### Experiment Matrix
- Baseline: YOLO fine‑tune on labeled only.
- SSL‑Curated: YOLO fine‑tune on labeled + curated subset (no pseudo‑labels).
- SSL‑Pseudo: YOLO fine‑tune on labeled + filtered pseudo‑labels.
- Ablations: with/without mosaic; with/without mixup; different unlabeled set sizes.

## Deliverables
- `scripts/build_dataset.py`: merge multiple roots, parse DataMuro, export unified dataset in requested format.
- `scripts/distill.py`: run Lightly Train (or local) distillation from DINOv3 → YOLOv12 backbone on unlabeled data, saving student backbone weights.
- `scripts/ssl_pretrain.py` (optional): DINOv3 SSL pretraining/embeddings for ablations or curation.
- `scripts/train.py`: fine‑tune Ultralytics YOLO using the unified dataset and optional SSL artifacts.
- `scripts/test.py`: evaluate a trained checkpoint and emit metrics/artifacts.
- `configs/*.yaml`: model/training/dataset configuration files.
- `docs/README.md` updates: quickstart and examples.

## Scripts and CLI Requirements

### Common Conventions
- All scripts accept `--manifest` to point to the YAML file and `--run-id` to override.
- Respect `--seed`, `--device` (e.g., `cpu`, `cuda:0`), deterministic flags.
- Support Windows paths and UNC shares.

### Dataset Builder
`scripts/build_dataset.py`

Required:
- `--manifest PATH` (required)

Optional:
- `--output-dir PATH` (default: `data/processed/<run_id>`)
- `--export-format {coco,ultralytics,datamuro}` (default from manifest)
- `--max-images INT` (debugging)

Outputs:
- Unified images folder and labels in the chosen export format.
- Mapping files, class list, split files.

Example:
```bash
python scripts/build_dataset.py --manifest E:/source/repos/dinov3_exp/manifests/dice.yaml --export-format coco
```

### Distillation (Lightly Train)
`scripts/distill.py`

Required:
- `--teacher {dinov3_s,dinov3_b}`
- `--student {yolov12n,yolov12s,yolov12m,yolov12l}`
- `--unlabeled-dirs PATH [PATH ...]` or `--manifest PATH`

Optional:
- `--epochs INT`, `--batch-size INT`, `--imgsz INT`
- `--feature-layers LIST` (student/teacher layer indices to align)
- `--loss {cosine,l2}` and `--loss-weight FLOAT`
- `--output-dir PATH` (default: `runs/distill/<run_id>`)
- Lightly Train integration: `--lightly-project STR`, `--lightly-dataset STR`

Outputs:
- Distilled YOLOv12 backbone weights `student_backbone.pt` and training logs.

### Training
`scripts/train.py`

Required:
- `--data PATH` (path to unified dataset directory or an Ultralytics data YAML)

Optional:
- `--model yolov12n` (or latest available, e.g., `yolov11n`, `yolov8n`)
- `--epochs INT`, `--imgsz INT`, `--batch INT`
- `--pretrained PATH_OR_NAME` (Ultralytics hub name or local `.pt` checkpoint)
- `--backbone-weights PATH` (path to distilled `student_backbone.pt` to initialize YOLO backbone)
- `--ssl-embeddings PATH` (optional; for curation ablations)
- `--pseudo-labels PATH` (optional; for semi‑supervised ablations)
- `--project PATH` and `--name STR` (Ultralytics run naming)

Outputs:
- YOLO run directory with checkpoints, metrics, plots.

Example:
```bash
python scripts/train.py --data data/processed/dice_exp_001/coco.yaml --model yolov12n --epochs 100 --imgsz 640
```

### Testing / Evaluation
`scripts/test.py`

Required:
- `--data PATH`
- `--weights PATH`

Optional:
- `--imgsz INT`, `--batch INT`
- `--save-json` (COCO results), `--plots`

Outputs:
- mAP@[.5:.95], precision, recall, PR curves, confusion matrix, per‑class metrics.

Example:
```bash
python scripts/test.py --data data/processed/dice_exp_001/coco.yaml --weights runs/detect/train/weights/best.pt --plots
```

## Metrics and Acceptance Criteria

### Metrics
- Primary: mAP@[.5:.95] on test split, mAP@.5, precision, recall.
- Efficiency: training time per epoch, GPU memory, throughput (img/s), inference latency (CPU/GPU) at `imgsz` 640.
- Data efficiency: performance vs. number of labeled images, and vs. percentage of unlabeled used.

### Acceptance Criteria (MVP)
- Scripts run on Windows with CUDA when available.
- Baseline model trains and achieves mAP@.5 ≥ 0.80 on test split (placeholder target; to be refined with real data).
- Distillation pipeline runs end‑to‑end (produces `student_backbone.pt` and integrates into YOLO training) and shows ≥ +2 points mAP@.5 over baseline on validation, or comparable accuracy with ≥ 20% fewer labeled images.

## Milestones
1) Week 1: Dataset builder + DataMuro adapter; baseline YOLO fine‑tune end‑to‑end.
2) Week 2: Distillation pipeline (Lightly Train): DINOv3 → YOLOv12 backbone on unlabeled; integrate backbone; evaluate vs baseline.
3) Week 3: Optional curation/pseudo‑label ablations; extended evaluations and sensitivity to unlabeled size.
4) Week 4: Stretch goal backbone/init refinements; documentation polish.

## Risks and Mitigations
- Distillation alignment between ViT (teacher) and YOLO backbone (student): use projection heads and resolution alignment; if unstable, reduce aligned layers or switch to cosine distance.
- DINOv3 → YOLO backbone incompatibility/version drift: fallback to representation‑level usage (curation/pseudo‑labels) and/or DINOv2.
- DataMuro format variance: define a strict adapter contract and unit tests; allow export to COCO to reduce risk.
- Multiple roots with duplicates: hash‑based deduplication and split consistency guards.
- Windows path edge cases/UNC shares: normalize paths; avoid symlinks; long path support.

## Environment and Dependencies
- Python 3.10+
- Ultralytics (latest), PyTorch w/ CUDA, torchvision
- Lightly (for curation/sampling/SSL utilities)
- Optional DINOv3 implementation (public weights or compatible repo)
- Additional: `opencv-python`, `numpy`, `PyYAML`, `tqdm`, `albumentations`

## Reproducibility
- Global seeding for data splits and dataloaders.
- Versioned configs under `configs/` checked into VCS.
- Save exact packages via `requirements.txt` and `pip freeze > runs/<run>/env.txt`.

## Appendix

### DataMuro Adapter Contract (expected)
- Input: directory with images and a manifest/annotation file(s) describing bounding boxes and class IDs.
- Output: internal representation normalized to `[x_min, y_min, width, height]` in pixels and class index; allow export to COCO JSON and Ultralytics YAML.

### Directory Structure (proposed)
```
E:/source/repos/dinov3_exp/
  configs/
  docs/
    PRD.md
  manifests/
    dice.yaml
  scripts/
    build_dataset.py
    ssl_pretrain.py
    train.py
    test.py
  data/
    processed/
      dice_exp_001/
        images/{train,val,test}
        labels/{train,val,test}
        coco.{train,val,test}.json
        coco.yaml
```

### Example Quickstart
```bash
# 1) Build dataset
python scripts/build_dataset.py --manifest manifests/dice.yaml --export-format coco

# 2) Distill DINOv3 → YOLOv12 backbone on unlabeled
python scripts/distill.py --manifest manifests/dice.yaml --teacher dinov3_b --student yolov12n --epochs 50 --imgsz 640

# 3) Train YOLO with distilled backbone
python scripts/train.py --data data/processed/dice_exp_001/coco.yaml --model yolov12n --backbone-weights runs/distill/dice_exp_001/student_backbone.pt --epochs 100 --imgsz 640

# 4) Test
python scripts/test.py --data data/processed/dice_exp_001/coco.yaml --weights runs/detect/train/weights/best.pt --plots
```


