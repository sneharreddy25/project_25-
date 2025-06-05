import cv2
import torch
import argparse
from pathlib import Path
import os
from src.detect import DefectDetector
import matplotlib.pyplot as plt
import numpy as np

def predict_image(detector, image_path, output_dir, confidence_threshold=0.5, show_all_scores=True):
    """Process a single image and save results."""
    # Read image
    image_path = str(Path(image_path))  # Convert to proper path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess image and get raw predictions
    preprocessed = detector.preprocess_image(image_rgb)
    preprocessed = preprocessed.to(detector.device)
    
    with torch.no_grad():
        # Get model predictions
        class_scores, bbox_pred = detector.model(preprocessed)
        probabilities = torch.softmax(class_scores, dim=1)
        
        # Get raw scores
        defect_probs = probabilities[:, 1]  # Probability of defect class
        normal_probs = probabilities[:, 0]  # Probability of normal class
        
        print("\nRaw Prediction Scores:")
        print(f"Normal probability: {normal_probs.item():.4f}")
        print(f"Defect probability: {defect_probs.item():.4f}")
    
    # Get detections with current threshold
    detections = detector.detect(image_rgb)
    
    # Draw results
    output_image = image.copy()
    if len(detections) > 0:
        for det in detections:
            # Get bounding box coordinates
            x1, y1, x2, y2 = [int(coord * image.shape[1 if i % 2 == 0 else 0]) for i, coord in enumerate(det['bbox'])]
            
            # Draw rectangle
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add label with confidence
            label = f"Defect: {det['confidence']:.2f}"
            cv2.putText(output_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Add score overlay
    score_text = f"Normal: {normal_probs.item():.2f} | Defect: {defect_probs.item():.2f}"
    cv2.putText(output_image, score_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save result
    output_path = Path(output_dir) / f"result_{Path(image_path).name}"
    cv2.imwrite(str(output_path), output_image)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Result image
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Defects (threshold={confidence_threshold})')
    plt.axis('off')
    
    # Probability plot
    plt.subplot(1, 3, 3)
    classes = ['Normal', 'Defect']
    probs = [normal_probs.item(), defect_probs.item()]
    plt.bar(classes, probs)
    plt.title('Class Probabilities')
    plt.ylim(0, 1)
    for i, v in enumerate(probs):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return detections

def main():
    parser = argparse.ArgumentParser(description='Run defect detection on images')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--confidence_threshold', type=float, default=0.3,
                      help='Confidence threshold for detections')
    parser.add_argument('--show_all_scores', action='store_true',
                      help='Show all prediction scores')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize detector
    detector = DefectDetector(
        model_path=args.model_path,
        confidence_threshold=args.confidence_threshold
    )
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Single image
        print(f"\nProcessing image: {input_path}")
        detections = predict_image(
            detector, 
            input_path, 
            output_dir,
            args.confidence_threshold,
            args.show_all_scores
        )
        print(f"\nDetections for {input_path.name}:")
        if len(detections) > 0:
            for det in detections:
                print(f"- Defect found with confidence: {det['confidence']:.4f}")
        else:
            print("No defects detected above threshold")
    else:
        # Directory of images
        image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
        print(f"\nFound {len(image_files)} images to process...")
        
        for img_path in image_files:
            print(f"\nProcessing {img_path.name}...")
            try:
                detections = predict_image(
                    detector, 
                    img_path, 
                    output_dir,
                    args.confidence_threshold,
                    args.show_all_scores
                )
                if len(detections) > 0:
                    print(f"Found {len(detections)} defects")
                else:
                    print("No defects detected above threshold")
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == '__main__':
    main() 