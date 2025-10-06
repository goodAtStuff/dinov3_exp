"""
Training script for YOLO detector.

Fine-tunes Ultralytics YOLO on dice detection dataset,
with optional support for loading distilled backbone weights.
"""

import argparse
import sys
from pathlib import Path
import logging
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger, load_yaml, normalize_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train YOLO detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset YAML file'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Model name or path to model checkpoint'
    )
    
    parser.add_argument(
        '--backbone-weights',
        type=str,
        default=None,
        help='Path to distilled backbone weights (optional)'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device (cuda:0, cpu, or empty for auto)'
    )
    
    # Optimizer settings
    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        choices=['auto', 'SGD', 'Adam', 'AdamW'],
        help='Optimizer'
    )
    
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience'
    )
    
    # Data augmentation
    parser.add_argument(
        '--mosaic',
        type=float,
        default=1.0,
        help='Mosaic augmentation probability'
    )
    
    parser.add_argument(
        '--mixup',
        type=float,
        default=0.0,
        help='Mixup augmentation probability'
    )
    
    # Output settings
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='train',
        help='Experiment name'
    )
    
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='Overwrite existing experiment'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to training config YAML (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML."""
    if config_path and config_path.exists():
        return load_yaml(config_path)
    return {}


def merge_args_with_config(args, config: dict) -> dict:
    """Merge command line args with config file."""
    # Start with config
    train_kwargs = config.get('training', {}).copy()
    aug_kwargs = config.get('augmentation', {}).copy()
    
    # Override with command line args
    train_kwargs.update({
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device if args.device else '',
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'patience': args.patience,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'resume': args.resume,
        'verbose': args.verbose,
    })
    
    # Merge augmentation settings
    train_kwargs.update({
        'mosaic': args.mosaic,
        'mixup': args.mixup,
    })
    
    # Add other augmentation settings from config
    train_kwargs.update(aug_kwargs)
    
    return train_kwargs


def load_backbone_weights(model, backbone_weights_path: Path, logger):
    """
    Load distilled backbone weights into YOLO model.
    
    This is a placeholder - actual implementation depends on
    the distillation output format and YOLO model structure.
    """
    logger.warning("Backbone weight loading is not fully implemented yet.")
    logger.warning("This requires matching layer names between distilled backbone and YOLO.")
    
    # Placeholder for actual implementation
    # try:
    #     import torch
    #     state_dict = torch.load(backbone_weights_path)
    #     # Match and load backbone layers
    #     model.model.load_state_dict(state_dict, strict=False)
    #     logger.info(f"Loaded backbone weights from {backbone_weights_path}")
    # except Exception as e:
    #     logger.error(f"Failed to load backbone weights: {e}")
    
    return model


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('train', level=log_level, console=True)
    
    logger.info("=" * 80)
    logger.info("YOLO Training")
    logger.info("=" * 80)
    
    try:
        # Load config if provided
        config = {}
        if args.config:
            config_path = normalize_path(args.config)
            logger.info(f"Loading config from {config_path}")
            config = load_config(config_path)
        
        # Merge arguments
        train_kwargs = merge_args_with_config(args, config)
        
        logger.info(f"Data: {args.data}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch}")
        logger.info(f"Image size: {args.imgsz}")
        logger.info(f"Device: {args.device if args.device else 'auto'}")
        
        # Import YOLO
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            sys.exit(1)
        
        # Load model
        logger.info(f"Loading model: {args.model}")
        model = YOLO(args.model)
        
        # Load backbone weights if provided
        if args.backbone_weights:
            backbone_path = normalize_path(args.backbone_weights)
            if backbone_path.exists():
                logger.info(f"Loading distilled backbone weights: {backbone_path}")
                model = load_backbone_weights(model, backbone_path, logger)
            else:
                logger.warning(f"Backbone weights not found: {backbone_path}")
        
        # Train model
        logger.info("Starting training...")
        logger.info(f"Training config: {train_kwargs}")
        
        results = model.train(**train_kwargs)
        
        # Log results
        logger.info("\n" + "=" * 80)
        logger.info("Training complete!")
        logger.info("=" * 80)
        
        # Get best model path
        best_model = Path(args.project) / args.name / 'weights' / 'best.pt'
        if best_model.exists():
            logger.info(f"Best model saved to: {best_model}")
        
        last_model = Path(args.project) / args.name / 'weights' / 'last.pt'
        if last_model.exists():
            logger.info(f"Last model saved to: {last_model}")
        
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Test model: python scripts/test.py --data {args.data} --weights {best_model}")
        logger.info(f"  2. Or continue training: python scripts/train.py --data {args.data} --model {last_model} --resume")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

