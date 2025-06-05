import torch
import numpy as np
from pathlib import Path
import json
import argparse
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime

from src.detect import DefectDetector
from src.utils import load_image, DefectLogger

def plot_precision_recall_curve(precisions, recalls, average_precision, output_dir):
    """Plot precision-recall curve."""
    plt.figure()
    plt.plot(recalls, precisions, color='darkorange', lw=2,
             label=f'AP = {average_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(output_dir / 'precision_recall_curve.png')
    plt.close()

def plot_confusion_matrix(cm, classes, output_dir):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()

def evaluate_model(detector, test_dir, output_dir):
    """Evaluate model performance on test set."""
    # Load test annotations
    with open(test_dir / 'annotations.json', 'r') as f:
        annotations = json.load(f)
    
    # Setup metrics
    y_true = []
    y_pred = []
    y_scores = []
    bbox_ious = []
    inference_times = []
    logger = DefectLogger()
    
    # Process each test image
    for ann in tqdm(annotations, desc='Evaluating'):
        image_path = test_dir / ann['image_name']
        
        # Load and process image
        image = load_image(str(image_path))
        start_time = datetime.now()
        detections = detector.detect(image)
        inference_time = (datetime.now() - start_time).total_seconds()
        inference_times.append(inference_time)
        
        # Get ground truth
        true_label = 1 if ann['has_defect'] else 0
        y_true.append(true_label)
        
        # Get predictions
        if detections:
            # Take highest confidence detection
            max_conf_detection = max(detections, key=lambda x: x['confidence'])
            y_pred.append(1)
            y_scores.append(max_conf_detection['confidence'])
            
            # Calculate IoU if bbox available
            if 'bbox' in ann:
                iou = calculate_iou(
                    max_conf_detection['bbox'],
                    ann['bbox']
                )
                bbox_ious.append(iou)
            
            # Log detection
            logger.log_defect(
                defect_type='visual_defect',
                confidence=max_conf_detection['confidence'],
                timestamp=datetime.now()
            )
        else:
            y_pred.append(0)
            y_scores.append(0.0)
        
        logger.total_inspections += 1
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    class_report = classification_report(y_true, y_pred, target_names=['Normal', 'Defect'])
    
    # Plot results
    plot_precision_recall_curve(precisions, recalls, average_precision, output_dir)
    plot_confusion_matrix(cm, ['Normal', 'Defect'], output_dir)
    
    # Calculate timing statistics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    # Calculate bbox accuracy if available
    bbox_accuracy = np.mean(bbox_ious) if bbox_ious else None
    
    # Save results
    results = {
        'average_precision': float(average_precision),
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'avg_inference_time': float(avg_inference_time),
        'std_inference_time': float(std_inference_time),
        'bbox_accuracy': float(bbox_accuracy) if bbox_accuracy is not None else None,
        'total_images': len(annotations),
        'defect_statistics': logger.get_statistics()
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Average Precision: {average_precision:.3f}")
    print(f"\nClassification Report:\n{class_report}")
    print(f"\nAverage Inference Time: {avg_inference_time*1000:.2f}ms ± {std_inference_time*1000:.2f}ms")
    if bbox_accuracy is not None:
        print(f"Average Bounding Box IoU: {bbox_accuracy:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate defect detection model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--test_dir', type=str, required=True,
                      help='Path to test dataset directory')
    parser.add_argument('--output_dir', type=str, default='evaluation',
                      help='Directory to save evaluation results')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                      help='Confidence threshold for detections')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize detector
    detector = DefectDetector(
        model_path=args.model_path,
        confidence_threshold=args.confidence_threshold
    )
    
    # Run evaluation
    evaluate_model(detector, Path(args.test_dir), output_dir)

if __name__ == '__main__':
    main() 