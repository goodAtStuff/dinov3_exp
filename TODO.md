## TODO: Dice Detector Project (DINOv3 → YOLOv12 Distillation)

### Dependencies and scaffolding
- [ ] Create `requirements.txt` including Ultralytics, Lightly, PyTorch CUDA, torchvision, opencv-python, numpy, pyyaml, tqdm, albumentations.
- [ ] Add `manifests/dice.yaml` template covering labeled/unlabeled roots, splits, and export format.
- [ ] Create `configs/` with default YAMLs: `dataset.yaml`, `distill.yaml`, `train.yaml`, `test.yaml`.

### Data ingestion and processing
- [ ] Implement `DataMuroAdapter` to load DataMuro annotations into an internal schema.
- [ ] Implement dataset merger to read multiple roots, normalize paths, and deduplicate by image hash.
- [ ] Implement deterministic splitter with seed; prevent leakage for duplicate hashes.
- [ ] Implement COCO export and an Ultralytics data YAML writer (`coco.yaml` with train/val/test paths).
- [ ] Add utilities: path normalization, image hashing, YAML utilities, logging helpers.

### Dataset builder script
- [ ] Implement `scripts/build_dataset.py` CLI and workflow (args: `--manifest`, `--output-dir`, `--export-format`, `--max-images`, `--seed`).
- [ ] Produce unified directory, COCO JSONs, `coco.yaml`, split manifests, class list; add logging and dry-run.

### Distillation pipeline (Lightly Train)
- [ ] Implement DINOv3 teacher loader with configurable variant (s/b), preprocessing, and feature extraction hooks.
- [ ] Implement YOLOv12 student backbone access, layer mapping, and projection heads for feature alignment.
- [ ] Implement distillation losses (cosine, L2) and layer-wise weighting; AMP + gradient clipping.
- [ ] Integrate Lightly Train session/config hooks; support local mode if Lightly Cloud unavailable.
- [ ] Save `runs/distill/<run_id>/student_backbone.pt` plus training logs and config snapshot.
- [ ] Implement `scripts/distill.py` CLI (args: `--teacher`, `--student`, `--manifest|--unlabeled-dirs`, `--epochs`, `--batch-size`, `--imgsz`, `--feature-layers`, `--loss`, `--loss-weight`, `--output-dir`, `--lightly-project`, `--lightly-dataset`).

### YOLO training
- [ ] Implement `scripts/train.py` using Ultralytics with support for `--backbone-weights` initialization.
- [ ] Wire dataset YAML (`coco.yaml`), epochs/imgsz/batch, project/name, resume/early-stop; log metrics and export best weights.

### Evaluation
- [ ] Implement `scripts/test.py` (args: `--data`, `--weights`, `--imgsz`, `--batch`, `--save-json`, `--plots`).
- [ ] Output mAP@[.5:.95], PR curves, confusion matrix, per-class metrics.

### Documentation and checks
- [ ] Update `docs/README.md` Quickstart aligned to PRD (build → distill → train → test).
- [ ] Add `scripts/env_check.py` to assert CUDA, versions, and path access.
- [ ] Add unit tests for `DataMuroAdapter`, COCO export, and split determinism.


