"""
Environment check script.

Verifies that all required dependencies are installed and
CUDA is available if requested.
"""

import sys
import platform
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python_version():
    """Check Python version."""
    print("=" * 80)
    print("Python Environment Check")
    print("=" * 80)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        return False
    else:
        print("✓ Python version OK")
        return True


def check_pytorch():
    """Check PyTorch installation and CUDA."""
    print("\n" + "=" * 80)
    print("PyTorch")
    print("=" * 80)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"✓ PyTorch installed")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            
            # Test CUDA
            try:
                x = torch.randn(10, 10).cuda()
                print(f"✓ CUDA test successful")
            except Exception as e:
                print(f"❌ CUDA test failed: {e}")
                return False
        else:
            print("⚠ CUDA not available (will use CPU)")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False


def check_package(name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
        return True
    except ImportError:
        print(f"❌ {name}: not installed")
        return False


def check_dependencies():
    """Check all required dependencies."""
    print("\n" + "=" * 80)
    print("Dependencies")
    print("=" * 80)
    
    packages = [
        ('Ultralytics', 'ultralytics'),
        ('Lightly', 'lightly'),
        ('OpenCV', 'cv2'),
        ('NumPy', 'numpy'),
        ('Pillow', 'PIL'),
        ('PyYAML', 'yaml'),
        ('tqdm', 'tqdm'),
        ('Albumentations', 'albumentations'),
        ('pycocotools', 'pycocotools'),
        ('pandas', 'pandas'),
        ('tensorboard', 'tensorboard'),
    ]
    
    all_ok = True
    for name, import_name in packages:
        ok = check_package(name, import_name)
        if not ok:
            all_ok = False
    
    return all_ok


def check_paths():
    """Check important project paths."""
    print("\n" + "=" * 80)
    print("Project Structure")
    print("=" * 80)
    
    root = Path(__file__).parent.parent
    
    paths_to_check = [
        ('configs/', 'Config directory'),
        ('manifests/', 'Manifests directory'),
        ('src/', 'Source code'),
        ('scripts/', 'Scripts directory'),
        ('requirements.txt', 'Requirements file'),
    ]
    
    all_ok = True
    for path, description in paths_to_check:
        full_path = root / path
        if full_path.exists():
            print(f"✓ {description}: {full_path}")
        else:
            print(f"❌ {description} not found: {full_path}")
            all_ok = False
    
    # Check data directory
    data_dir = root / 'data'
    if data_dir.exists():
        print(f"✓ Data directory: {data_dir}")
    else:
        print(f"⚠ Data directory not found: {data_dir} (will be created when needed)")
    
    return all_ok


def check_system():
    """Check system information."""
    print("\n" + "=" * 80)
    print("System Information")
    print("=" * 80)
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"RAM: {mem.total / (1024**3):.2f} GB total, "
              f"{mem.available / (1024**3):.2f} GB available")
    except ImportError:
        print("⚠ Install psutil for memory info: pip install psutil")
    
    return True


def main():
    """Run all checks."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DINOv3 Experiment Environment Check" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    checks = [
        ("Python version", check_python_version),
        ("PyTorch", check_pytorch),
        ("Dependencies", check_dependencies),
        ("Project structure", check_paths),
        ("System info", check_system),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print("✓ All checks passed! Environment is ready.")
        print("\nNext steps:")
        print("  1. Create your dataset manifest in manifests/")
        print("  2. Build dataset: python scripts/build_dataset.py --manifest manifests/dice.yaml")
        print("  3. Train model: python scripts/train.py --data <dataset_yaml>")
        return 0
    else:
        print("❌ Some checks failed. Please install missing dependencies.")
        print("\nTo install all dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())

