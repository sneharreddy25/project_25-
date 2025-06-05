import os
import urllib.request
import zipfile
import json
from pathlib import Path
import shutil
import random
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        def report_hook(count, block_size, total_size):
            t.total = total_size
            t.update(block_size)
        urllib.request.urlretrieve(url, filename, reporthook=report_hook)

def create_dataset_structure():
    """Create the required directory structure"""
    for split in ['train', 'val', 'test']:
        os.makedirs(f'dataset/{split}/images', exist_ok=True)

def process_mvtec_data(category):
    """Process MVTec dataset for a specific category"""
    # Download dataset
    url = f'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/{category}.tar.xz'
    filename = f'{category}.tar.xz'
    print(f'Downloading {category} dataset...')
    download_file(url, filename)
    
    # Extract dataset
    print(f'Extracting {category} dataset...')
    os.system(f'tar xf {filename}')
    
    # Process images and create annotations
    train_annotations = []
    val_annotations = []
    test_annotations = []
    
    # Process good (non-defective) images
    good_images = list(Path(category, 'train', 'good').glob('*.png'))
    random.shuffle(good_images)
    
    # Split good images
    n_good = len(good_images)
    train_good = good_images[:int(0.7 * n_good)]
    val_good = good_images[int(0.7 * n_good):int(0.85 * n_good)]
    test_good = good_images[int(0.85 * n_good):]
    
    # Process defective images
    defect_types = [d for d in os.listdir(os.path.join(category, 'test')) 
                   if d != 'good' and os.path.isdir(os.path.join(category, 'test', d))]
    
    defect_images = []
    for defect_type in defect_types:
        defect_images.extend(list(Path(category, 'test', defect_type).glob('*.png')))
    random.shuffle(defect_images)
    
    # Split defective images
    n_defect = len(defect_images)
    train_defect = defect_images[:int(0.7 * n_defect)]
    val_defect = defect_images[int(0.7 * n_defect):int(0.85 * n_defect)]
    test_defect = defect_images[int(0.85 * n_defect):]
    
    # Copy and annotate training images
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
    with open('dataset/train/annotations.json', 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open('dataset/val/annotations.json', 'w') as f:
        json.dump(val_annotations, f, indent=2)
    
    with open('dataset/test/annotations.json', 'w') as f:
        json.dump(test_annotations, f, indent=2)
    
    # Cleanup
    os.remove(filename)
    shutil.rmtree(category)

def main():
    # Create dataset structure
    create_dataset_structure()
    
    # Process a specific category (e.g., 'bottle', 'cable', 'capsule', etc.)
    # We'll use 'bottle' as an example
    process_mvtec_data('bottle')
    
    print('Dataset preparation completed!')
    print('Dataset structure:')
    os.system('tree dataset')

if __name__ == '__main__':
    main() 