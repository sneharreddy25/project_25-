import cv2
import torch
import argparse
from pathlib import Path
import time
from src.detect import DefectDetector
import numpy as np

def process_video(detector, video_source, output_path=None, confidence_threshold=0.3):
    """Process video stream for defect detection.
    
    Args:
        detector: DefectDetector instance
        video_source: Integer for webcam or string path for video file
        output_path: Path to save processed video (optional)
        confidence_threshold: Detection confidence threshold
    """
    # Open video capture
    if isinstance(video_source, str):
        cap = cv2.VideoCapture(video_source)
    else:
        cap = cv2.VideoCapture(video_source)  # For webcam
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video source {video_source}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is provided
    writer = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Initialize performance metrics
    frame_count = 0
    total_time = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            process_start = time.time()
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get predictions
            preprocessed = detector.preprocess_image(frame_rgb)
            preprocessed = preprocessed.to(detector.device)
            
            with torch.no_grad():
                # Get model predictions
                class_scores, bbox_pred = detector.model(preprocessed)
                probabilities = torch.softmax(class_scores, dim=1)
                
                # Get scores
                defect_prob = probabilities[0, 1].item()
                normal_prob = probabilities[0, 0].item()
            
            # Get detections
            detections = detector.detect(frame_rgb)
            
            # Draw results
            output_frame = frame.copy()
            
            # Draw detections
            if len(detections) > 0:
                for det in detections:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = [int(coord * frame.shape[1 if i % 2 == 0 else 0]) 
                                    for i, coord in enumerate(det['bbox'])]
                    
                    # Draw rectangle
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Add label with confidence
                    label = f"Defect: {det['confidence']:.2f}"
                    cv2.putText(output_frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Calculate and display FPS
            process_time = time.time() - process_start
            total_time += process_time
            avg_fps = frame_count / (time.time() - start_time)
            
            # Add overlay information
            info_text = [
                f"FPS: {avg_fps:.1f}",
                f"Normal: {normal_prob:.2f}",
                f"Defect: {defect_prob:.2f}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(output_frame, text, (10, 30 + i*30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame if output path provided
            if writer:
                writer.write(output_frame)
            
            # Display frame
            cv2.imshow('Defect Detection', output_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print performance statistics
        avg_fps = frame_count / (time.time() - start_time)
        avg_process_time = total_time / frame_count
        print(f"\nProcessing complete:")
        print(f"Processed {frame_count} frames")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Average processing time per frame: {avg_process_time*1000:.1f}ms")

def main():
    parser = argparse.ArgumentParser(description='Run defect detection on video')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--input', type=str,
                      help='Path to input video file (optional, uses webcam if not provided)')
    parser.add_argument('--output', type=str,
                      help='Path to save output video (optional)')
    parser.add_argument('--confidence_threshold', type=float, default=0.3,
                      help='Confidence threshold for detections')
    parser.add_argument('--device', type=int, default=0,
                      help='Camera device number (default: 0)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DefectDetector(
        model_path=args.model_path,
        confidence_threshold=args.confidence_threshold
    )
    
    # Process video
    try:
        video_source = args.input if args.input else args.device
        process_video(detector, video_source, args.output, args.confidence_threshold)
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == '__main__':
    main() 