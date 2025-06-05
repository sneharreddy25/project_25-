import os
import json
from pathlib import Path
import shutil
import random

def create_dataset_structure():
    """Create the required directory structure"""
    for split in ['train', 'val', 'test']:
        os.makedirs(f'dataset/{split}/images', exist_ok=True)

def convert_mvtec_data(source_dir, category):
    """Convert MVTec dataset from a specific category into our format
    
    Args:
        source_dir: Path to the downloaded and extracted dataset directory
        category: Category name (e.g., 'bottle', 'cable', etc.)
    """
    category_dir = os.path.join(source_dir, category)
    if not os.path.exists(category_dir):
        raise ValueError(f"Category directory {category_dir} not found!")
    
    # Process images and create annotations
    train_annotations = []
    val_annotations = []
    test_annotations = []
    
    # Process good (non-defective) images
    good_images = list(Path(category_dir, 'train', 'good').glob('*.png'))
    random.shuffle(good_images)
    
    # Split good images
    n_good = len(good_images)
    train_good = good_images[:int(0.7 * n_good)]
    val_good = good_images[int(0.7 * n_good):int(0.85 * n_good)]
    test_good = good_images[int(0.85 * n_good):]
    
    # Process defective images
    defect_types = [d for d in os.listdir(os.path.join(category_dir, 'test')) 
                   if d != 'good' and os.path.isdir(os.path.join(category_dir, 'test', d))]
    
    defect_images = []
    for defect_type in defect_types:
        defect_images.extend(list(Path(category_dir, 'test', defect_type).glob('*.png')))
    random.shuffle(defect_images)
    
    # Split defective images
    n_defect = len(defect_images)
    train_defect = defect_images[:int(0.7 * n_defect)]
    val_defect = defect_images[int(0.7 * n_defect):int(0.85 * n_defect)]
    test_defect = defect_images[int(0.85 * n_defect):]
    
    print(f"\nProcessing {category} dataset:")
    print(f"Found {n_good} good images and {n_defect} defective images")
    print(f"Defect types: {', '.join(defect_types)}")
    
    # Copy and annotate training images
    print("\nProcessing training set...")
    for img in train_good:
        shutil.copy2(img, f'dataset/train/images/{img.name}')
        train_annotations.append({
            'image_name': img.name,
            'has_defect': False
        })
    
    for img in train_defect:
        shutil.copy2(img, f'dataset/train/images/{img.name}')
        train_annotations.append({
            'image_name': img.name,
            'has_defect': True,
            'defect_type': img.parent.name
        })
    
    # Copy and annotate validation images
    print("Processing validation set...")
    for img in val_good:
        shutil.copy2(img, f'dataset/val/images/{img.name}')
        val_annotations.append({
            'image_name': img.name,
            'has_defect': False
        })
    
    for img in val_defect:
        shutil.copy2(img, f'dataset/val/images/{img.name}')
        val_annotations.append({
            'image_name': img.name,
            'has_defect': True,
            'defect_type': img.parent.name
        })
    
    # Copy and annotate test images
    print("Processing test set...")
    for img in test_good:
        shutil.copy2(img, f'dataset/test/images/{img.name}')
        test_annotations.append({
            'image_name': img.name,
            'has_defect': False
        })
    
    for img in test_defect:
        shutil.copy2(img, f'dataset/test/images/{img.name}')
        test_annotations.append({
            'image_name': img.name,
            'has_defect': True,
            'defect_type': img.parent.name
        })
    
    # Save annotations
    print("\nSaving annotations...")
    with open('dataset/train/annotations.json', 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open('dataset/val/annotations.json', 'w') as f:
        json.dump(val_annotations, f, indent=2)
    
    with open('dataset/test/annotations.json', 'w') as f:
        json.dump(test_annotations, f, indent=2)
    
    print("\nDataset statistics:")
    print(f"Training set: {len(train_annotations)} images")
    print(f"Validation set: {len(val_annotations)} images")
    print(f"Test set: {len(test_annotations)} images")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert MVTec dataset to our format')
    parser.add_argument('--source_dir', type=str, required=True,
                      help='Path to the downloaded and extracted dataset directory')
    parser.add_argument('--category', type=str, required=True,
                      help='Category to process (e.g., bottle, cable, etc.)')
    
    args = parser.parse_args()
    
    # Create dataset structure
    create_dataset_structure()
    
    # Convert the dataset
    convert_mvtec_data(args.source_dir, args.category)
    
    print('\nDataset conversion completed!')
    print('Dataset structure:')
    os.system('tree dataset')

if __name__ == '__main__':
    main() 