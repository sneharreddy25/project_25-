import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
import time
from src.detect import DefectDetector
from src.utils import DefectLogger, create_defect_report

def process_image(detector: DefectDetector, image_path: str, output_dir: Path):
    """Process a single image and save results."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Get detections
    detections, annotated_frame = detector.process_video_frame(image)
    
    # Save results
    output_path = output_dir / f"result_{Path(image_path).name}"
    cv2.imwrite(str(output_path), annotated_frame)
    
    return detections

def process_video(detector: DefectDetector, video_source: int, output_dir: Path):
    """Process video stream and save results."""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source {video_source}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup video writer
    output_path = output_dir / "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Setup defect logger
    logger = DefectLogger()
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            detections, annotated_frame = detector.process_video_frame(frame)
            
            # Log detections
            for detection in detections:
                logger.log_defect(
                    defect_type='visual_defect',
                    confidence=detection['confidence'],
                    timestamp=time.time()
                )
            logger.total_inspections += 1
            
            # Add FPS info
            fps = frame_count / (time.time() - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2)
            
            # Write frame
            out.write(annotated_frame)
            
            # Display frame
            cv2.imshow('Defect Detection', annotated_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Create report
        create_defect_report(logger, output_dir / "report.txt")

def main():
    parser = argparse.ArgumentParser(description='Demo defect detection system')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--input', type=str,
                      help='Path to input image or video file')
    parser.add_argument('--camera', type=int, default=0,
                      help='Camera device number for live demo')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save results')
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
    
    if args.input and Path(args.input).is_file():
        # Process image/video file
        if Path(args.input).suffix.lower() in ['.jpg', '.jpeg', '.png']:
            process_image(detector, args.input, output_dir)
        else:
            process_video(detector, args.input, output_dir)
    else:
        # Process camera feed
        process_video(detector, args.camera, output_dir)

if __name__ == '__main__':
    main() 