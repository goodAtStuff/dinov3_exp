"""YAML file utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary with YAML contents
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        logger.debug(f"Loaded YAML from {file_path}")
        return data if data is not None else {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path],
             sort_keys: bool = False, indent: int = 2) -> None:
    """
    Save data to a YAML file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save to
        sort_keys: Whether to sort dictionary keys
        indent: Indentation level
    """
    file_path = Path(file_path)
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, sort_keys=sort_keys, indent=indent,
                          default_flow_style=False, allow_unicode=True)
        logger.debug(f"Saved YAML to {file_path}")
    except Exception as e:
        logger.error(f"Error writing YAML file {file_path}: {e}")
        raise


def load_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate a dataset manifest file.
    
    Args:
        manifest_path: Path to manifest YAML file
        
    Returns:
        Validated manifest dictionary
        
    Raises:
        ValueError: If manifest is invalid
    """
    manifest = load_yaml(manifest_path)
    
    # Validate required fields
    required_fields = ['run_id', 'classes', 'label_format']
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Manifest missing required field: {field}")
    
    # Validate roots structure
    if 'roots' not in manifest:
        raise ValueError("Manifest missing 'roots' section")
    
    roots = manifest['roots']
    if not isinstance(roots, dict):
        raise ValueError("Manifest 'roots' must be a dictionary")
    
    # Ensure at least labeled or unlabeled roots exist
    labeled_roots = roots.get('labeled', [])
    unlabeled_roots = roots.get('unlabeled', [])
    
    if not labeled_roots and not unlabeled_roots:
        raise ValueError("Manifest must specify at least one labeled or unlabeled root")
    
    # Validate splits if present
    if 'splits' in manifest:
        splits = manifest['splits']
        if 'train_ratio' in splits and 'val_ratio' in splits and 'test_ratio' in splits:
            total = splits['train_ratio'] + splits['val_ratio'] + splits['test_ratio']
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    # Set defaults
    if 'seed' not in manifest:
        manifest['seed'] = 42
        logger.info("Using default seed: 42")
    
    if 'export' not in manifest:
        manifest['export'] = {'format': 'coco'}
        logger.info("Using default export format: coco")
    
    logger.info(f"Loaded manifest for run_id: {manifest['run_id']}")
    
    return manifest


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Performs deep merge for nested dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override value
            result[key] = value
    
    return result

