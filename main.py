import cv2
import argparse
import time
from pathlib import Path
from ultralytics import YOLO


def format_time(seconds):
    """Convert seconds to human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def process_video_with_yolo(input_path, output_path, model_path="best-1.pt", conf_threshold=0.5):
    """
    Process a video file with YOLOv8 detection and save the output video.

    Args:
        input_path (str): Path to input video file
        output_path (str): Path for output video file
        model_path (str): Path to YOLO model file (default: yolov8n.pt)
        conf_threshold (float): Confidence threshold for detections
    """

    # Load YOLO model
    print(f"Loading YOLO model: {model_path}")
    model_load_start = time.time()
    model = YOLO(model_path)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f} seconds")

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {input_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Estimated video duration: {total_frames/fps:.1f} seconds")

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_inference_time = 0
    total_processing_time = 0
    start_time = time.time()

    # Progress tracking variables
    progress_update_interval = max(1, total_frames // 100)  # Update every 1% or at least every frame
    last_update_time = start_time

    print(f"\nStarting video processing...")
    print(f"Confidence threshold: {conf_threshold}")
    print("-" * 70)

    try:
        while True:
            frame_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run YOLO inference
            inference_start = time.time()
            results = model(frame, conf=conf_threshold, verbose=False)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            # Draw detections on frame
            annotated_frame = results[0].plot()

            # Write frame to output video
            out.write(annotated_frame)

            frame_processing_time = time.time() - frame_start_time
            total_processing_time += frame_processing_time

            # Progress update
            current_time = time.time()
            if frame_count % progress_update_interval == 0 or frame_count == total_frames:
                elapsed_time = current_time - start_time
                progress_percent = (frame_count / total_frames) * 100

                # Calculate speeds
                avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                avg_inference_time = total_inference_time / frame_count
                avg_processing_time = total_processing_time / frame_count

                # Estimate remaining time
                if avg_fps > 0:
                    remaining_frames = total_frames - frame_count
                    eta_seconds = remaining_frames / avg_fps
                    eta_formatted = format_time(eta_seconds)
                else:
                    eta_formatted = "Unknown"

                # Count detections in current frame
                num_detections = len(results[0].boxes) if results[0].boxes is not None else 0

                # Progress display
                elapsed_formatted = format_time(elapsed_time)
                print(f"Frame {frame_count:6d}/{total_frames} ({progress_percent:5.1f}%) | "
                      f"Speed: {avg_fps:5.1f} fps | "
                      f"Inference: {avg_inference_time*1000:5.1f}ms | "
                      f"Detections: {num_detections:2d} | "
                      f"Elapsed: {elapsed_formatted} | "
                      f"ETA: {eta_formatted}", end='\r')

    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("Processing interrupted by user")

    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Final statistics
        total_time = time.time() - start_time
        final_fps = frame_count / total_time if total_time > 0 else 0
        avg_inference_ms = (total_inference_time / frame_count * 1000) if frame_count > 0 else 0
        avg_processing_ms = (total_processing_time / frame_count * 1000) if frame_count > 0 else 0

        print("\n" + "="*70)
        print("PROCESSING COMPLETE!")
        print("="*70)
        print(f"Output saved to: {output_path}")
        print(f"Frames processed: {frame_count}/{total_frames}")
        print(f"Total time: {format_time(total_time)}")
        print(f"Average processing speed: {final_fps:.2f} fps")
        print(f"Average inference time: {avg_inference_ms:.1f} ms/frame")
        print(f"Average total processing time: {avg_processing_ms:.1f} ms/frame")
        print(f"Total inference time: {format_time(total_inference_time)}")

        # Efficiency metrics
        if total_time > 0:
            real_time_factor = (total_frames / fps) / total_time
            print(f"Real-time factor: {real_time_factor:.2f}x")
            if real_time_factor >= 1.0:
                print("✓ Processing faster than real-time!")
            else:
                print("⚠ Processing slower than real-time")


def process_video_with_yolo_batch(input_path, output_path, model_path="yolov8n.pt",
                                  conf_threshold=0.25, batch_size=1):

    print(f"Loading YOLO model: {model_path}")
    model_load_start = time.time()
    model = YOLO(model_path)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f} seconds")

    print(f"Starting batch processing with built-in YOLO functionality...")
    start_time = time.time()

    results = model.track(
        source=input_path,
        save=True,
        project=str(Path(output_path).parent),
        name=Path(output_path).stem,
        conf=conf_threshold,
        show=False,
        verbose=True
    )

    total_time = time.time() - start_time
    print(f"Batch processing complete in {format_time(total_time)}")
    print(f"Check the runs/detect folder for output")


def main():
    parser = argparse.ArgumentParser(description="Process video with YOLOv8 detection")
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("output", help="Path for output video file")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Path to YOLO model weights file (e.g., best.pt, your_model.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--method", choices=["manual", "batch"], default="manual",
                        help="Processing method: manual (frame-by-frame) or batch (built-in)")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' does not exist")
        return


    if not Path(args.model).exists():
        print(f"Error: Model file '{args.model}' does not exist")
        return

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Using custom trained model: {args.model}")
        if args.method == "manual":
            process_video_with_yolo(args.input, args.output, args.model, args.conf)
        else:
            process_video_with_yolo_batch(args.input, args.output, args.model, args.conf)

    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":

    input_video = "pigs.mp4"
    output_video = "output_video.mp4"
    custom_model = "best-1.pt"

    process_video_with_yolo(input_video, output_video, custom_model, conf_threshold=0.25)
