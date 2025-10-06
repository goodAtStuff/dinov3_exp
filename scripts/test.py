"""
Testing/evaluation script for YOLO detector.

Evaluates trained model on test dataset and generates metrics/plots.
"""

import argparse
import sys
from pathlib import Path
import logging
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger, load_yaml, normalize_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test/evaluate YOLO detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset YAML file'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained model weights'
    )
    
    # Test settings
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for inference'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.001,
        help='Confidence threshold'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.6,
        help='IoU threshold for NMS'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device (cuda:0, cpu, or empty for auto)'
    )
    
    # Output settings
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save results in COCO JSON format'
    )
    
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save results as text files'
    )
    
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate and save plots'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='test',
        help='Experiment name'
    )
    
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='Overwrite existing experiment'
    )
    
    # Performance profiling
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Profile inference speed'
    )
    
    parser.add_argument(
        '--profile-iterations',
        type=int,
        default=100,
        help='Number of iterations for profiling'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to test config YAML (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    return parser.parse_args()


def profile_inference(model, imgsz: int, iterations: int, 
                     device: str, logger) -> dict:
    """
    Profile model inference speed.
    
    Returns:
        Dictionary with latency and throughput metrics
    """
    logger.info(f"Profiling inference speed ({iterations} iterations)...")
    
    try:
        import torch
        import numpy as np
        
        # Create dummy input
        if 'cuda' in device and torch.cuda.is_available():
            device_obj = torch.device(device)
        else:
            device_obj = torch.device('cpu')
        
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device_obj)
        
        # Warmup
        logger.info("Warming up...")
        for _ in range(10):
            _ = model(dummy_input)
        
        # Profile
        logger.info("Profiling...")
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            times.append(end - start)
        
        times = np.array(times)
        
        metrics = {
            'mean_latency_ms': float(times.mean() * 1000),
            'std_latency_ms': float(times.std() * 1000),
            'min_latency_ms': float(times.min() * 1000),
            'max_latency_ms': float(times.max() * 1000),
            'throughput_fps': float(1.0 / times.mean()),
            'device': str(device_obj),
            'image_size': imgsz,
        }
        
        logger.info(f"Mean latency: {metrics['mean_latency_ms']:.2f} ± "
                   f"{metrics['std_latency_ms']:.2f} ms")
        logger.info(f"Throughput: {metrics['throughput_fps']:.2f} FPS")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        return {}


def main():
    """Main testing function."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('test', level=log_level, console=True)
    
    logger.info("=" * 80)
    logger.info("YOLO Testing/Evaluation")
    logger.info("=" * 80)
    
    try:
        # Validate inputs
        weights_path = normalize_path(args.weights)
        if not weights_path.exists():
            logger.error(f"Weights file not found: {weights_path}")
            sys.exit(1)
        
        data_path = normalize_path(args.data)
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            sys.exit(1)
        
        logger.info(f"Data: {data_path}")
        logger.info(f"Weights: {weights_path}")
        logger.info(f"Image size: {args.imgsz}")
        logger.info(f"Batch size: {args.batch}")
        logger.info(f"Confidence threshold: {args.conf}")
        logger.info(f"IoU threshold: {args.iou}")
        
        # Import YOLO
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            sys.exit(1)
        
        # Load model
        logger.info(f"Loading model from {weights_path}")
        model = YOLO(str(weights_path))
        
        # Profile inference if requested
        if args.profile:
            profile_metrics = profile_inference(
                model,
                args.imgsz,
                args.profile_iterations,
                args.device if args.device else 'cuda:0',
                logger
            )
            
            # Save profile results
            output_dir = Path(args.project) / args.name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            profile_path = output_dir / 'profile.json'
            with open(profile_path, 'w') as f:
                json.dump(profile_metrics, f, indent=2)
            logger.info(f"Saved profiling results to {profile_path}")
        
        # Run validation
        logger.info("\nRunning validation on test set...")
        
        val_kwargs = {
            'data': str(data_path),
            'imgsz': args.imgsz,
            'batch': args.batch,
            'conf': args.conf,
            'iou': args.iou,
            'device': args.device if args.device else '',
            'save_json': args.save_json,
            'save_txt': args.save_txt,
            'plots': args.plots,
            'project': args.project,
            'name': args.name,
            'exist_ok': args.exist_ok,
            'verbose': args.verbose,
        }
        
        results = model.val(**val_kwargs)
        
        # Log results
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Results")
        logger.info("=" * 80)
        
        # Extract key metrics
        if hasattr(results, 'box'):
            box_metrics = results.box
            logger.info(f"mAP50: {box_metrics.map50:.4f}")
            logger.info(f"mAP50-95: {box_metrics.map:.4f}")
            logger.info(f"Precision: {box_metrics.mp:.4f}")
            logger.info(f"Recall: {box_metrics.mr:.4f}")
            
            # Save metrics to JSON
            output_dir = Path(args.project) / args.name
            metrics = {
                'mAP50': float(box_metrics.map50),
                'mAP50-95': float(box_metrics.map),
                'precision': float(box_metrics.mp),
                'recall': float(box_metrics.mr),
            }
            
            metrics_path = output_dir / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"\nSaved metrics to {metrics_path}")
        
        output_dir = Path(args.project) / args.name
        logger.info(f"Results saved to: {output_dir}")
        
        if args.plots:
            logger.info(f"Plots saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

