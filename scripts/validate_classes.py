"""
Validate dice class mapping.

Run this to verify that the 71-class system is working correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dice_classes import (
    CLASS_NAMES,
    get_class_id,
    get_class_name,
    get_die_info_from_class_id,
    validate_class_mapping,
    get_class_statistics
)


def main():
    """Run validation."""
    print("="*80)
    print("Dice Class Mapping Validation")
    print("="*80)
    print()
    
    # Validate mapping
    validate_class_mapping()
    print()
    
    # Show statistics
    print("="*80)
    print("Class Statistics")
    print("="*80)
    stats = get_class_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Show sample classes
    print("="*80)
    print("Sample Class Mappings")
    print("="*80)
    
    samples = [
        (0, "Background"),
        (1, "First die class"),
        (10, "d6_6"),
        (28, "d10_10 (also covers d10 showing 0)"),
        (40, "d12_12"),
        (60, "d20_20"),
        (70, "d100_100 (also covers d100 showing 0)"),
    ]
    
    for class_id, description in samples:
        if class_id < len(CLASS_NAMES):
            print(f"  Class {class_id:2d}: {CLASS_NAMES[class_id]:15s} - {description}")
    print()
    
    # Test special cases
    print("="*80)
    print("Special Cases (d10 and d100 with value 0)")
    print("="*80)
    
    test_cases = [
        ('d10', 0, 'd10_10'),
        ('d10', 10, 'd10_10'),
        ('d100', 0, 'd100_100'),
        ('d100', 100, 'd100_100'),
    ]
    
    for die_type, value, expected_name in test_cases:
        class_id = get_class_id(die_type, value)
        class_name = get_class_name(die_type, value)
        print(f"  {die_type} value {value:3d} → class_id={class_id:2d}, "
              f"class_name={class_name:12s} (expected: {expected_name})")
        
        if class_name != expected_name:
            print(f"    ❌ ERROR: Expected {expected_name}, got {class_name}")
        else:
            print(f"    ✓ PASS")
    print()
    
    # Test all die types
    print("="*80)
    print("All Die Types and Values")
    print("="*80)
    
    die_tests = [
        ('d4', [1, 2, 3, 4]),
        ('d6', [1, 2, 3, 4, 5, 6]),
        ('d8', [1, 2, 3, 4, 5, 6, 7, 8]),
        ('d10', [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10]),  # Test both 0 and 10
        ('d12', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        ('d20', [1, 5, 10, 15, 20]),  # Sample values
        ('d100', [10, 20, 30, 40, 50, 60, 70, 80, 90, 0, 100]),  # Test both 0 and 100
    ]
    
    for die_type, values in die_tests:
        print(f"\n  {die_type}:")
        for value in values:
            class_id = get_class_id(die_type, value)
            class_name = get_class_name(die_type, value)
            if class_id is not None:
                print(f"    {die_type} = {value:3d} → class {class_id:2d} ({class_name})")
            else:
                print(f"    {die_type} = {value:3d} → ERROR: Could not map to class")
    print()
    
    # Reverse mapping test
    print("="*80)
    print("Reverse Mapping (Class ID → Die Info)")
    print("="*80)
    
    test_ids = [0, 1, 5, 11, 19, 29, 41, 61, 70]
    for class_id in test_ids:
        die_info = get_die_info_from_class_id(class_id)
        if die_info:
            die_type, value = die_info
            print(f"  Class {class_id:2d} → {die_type} value {value}")
        else:
            print(f"  Class {class_id:2d} → Background")
    print()
    
    print("="*80)
    print("✓ Validation Complete!")
    print("="*80)
    print(f"\nTotal classes: {len(CLASS_NAMES)}")
    print(f"Ready to use with manifest: classes: auto")


if __name__ == '__main__':
    main()

