A Python application for processing videos with custom YOLOv8 models. Features real-time progress tracking, performance metrics, ETA estimation, and flexible processing modes. Perfect for surveillance, sports analysis, or any video inference tasks.
```

---

# YOLOv8 Video Processor

A comprehensive Python application for processing videos with YOLOv8 object detection models, featuring real-time progress tracking, detailed performance metrics, and support for custom trained models.

## 🎯 Features

- **Custom Model Support**: Compatible with your own trained YOLOv8 models (.pt files)
- **Real-time Progress Tracking**: Live updates showing frame progress, processing speed, and ETA
- **Detailed Performance Metrics**: Inference timing, detection counts, and efficiency analysis
- **Flexible Processing Methods**: Choose between frame-by-frame or batch processing
- **Command Line Interface**: Easy-to-use CLI with customizable parameters
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 🚀 Quick Start

### Installation

1. **Create a conda environment (recommended):**
```bash
conda create -n yolo-video python=3.8
conda activate yolo-video
```

2. **Install dependencies:**
```bash
pip install ultralytics==8.3.180 opencv-python
```

3. **Clone the repository:**
```bash
git clone https://github.com/yourusername/yolov8-video-processor.git
cd yolov8-video-processor
```

### Basic Usage

```bash
# Process video with your custom model
python yolo_video_processor.py input_video.mp4 output_video.mp4 --model your_model.pt --conf 0.5

# Frame-by-frame processing with detailed progress
python yolo_video_processor.py video.mp4 output.mp4 --model best.pt --conf 0.6 --method manual

# Batch processing (faster, less control)
python yolo_video_processor.py video.mp4 output.mp4 --model best.pt --method batch
```

## 📊 Progress Display

The application provides comprehensive real-time feedback during processing:

```
Loading YOLO model: best-1.pt
Model loaded in 2.45 seconds
Video properties: 1920x1080 @ 30fps, 1500 frames
Estimated video duration: 50.0 seconds
----------------------------------------------------------------------
Frame   1234/1500 ( 82.3%) | Speed:  25.4 fps | Inference:  15.2ms | Detections:  3 | Elapsed: 01:28 | ETA: 00:19
```

## 📈 Performance Summary

After processing, get detailed statistics:

```
======================================================================
PROCESSING COMPLETE!
======================================================================
Output saved to: output_video.mp4
Frames processed: 1500/1500
Total time: 01:47
Average processing speed: 23.8 fps
Average inference time: 16.1 ms/frame
Real-time factor: 1.2x
✓ Processing faster than real-time!
```

## 🛠️ Command Line Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input` | Path to input video file | Required |
| `output` | Path for output video file | Required |
| `--model` | Path to YOLO model weights (.pt file) | `yolov8n.pt` |
| `--conf` | Confidence threshold (0.0-1.0) | `0.25` |
| `--method` | Processing method: `manual` or `batch` | `manual` |

## 💻 Python API Usage

```python
from yolo_video_processor import process_video_with_yolo

# Process video with custom settings
process_video_with_yolo(
    input_path="surveillance_footage.mp4",
    output_path="detected_objects.mp4",
    model_path="runs/detect/train/weights/best.pt",
    conf_threshold=0.6
)
```

## 🎯 Use Cases

- **Surveillance Analysis**: Process security camera footage for object/person detection
- **Sports Analytics**: Analyze game footage for player/ball tracking
- **Wildlife Monitoring**: Detect animals in nature documentaries or trail cam footage  
- **Traffic Analysis**: Monitor vehicle detection and counting
- **Quality Control**: Industrial inspection with custom trained models

## 📋 Requirements

- **Python**: 3.8 or higher
- **Dependencies**:
  - `ultralytics==8.3.180`
  - `opencv-python`
  - `pathlib` (built-in)
  - `argparse` (built-in)

## 🔧 Environment Setup

### Using Conda (Recommended)

```bash
# Create environment
conda create -n yolo-video python=3.8

# Activate environment
conda activate yolo-video

# Install packages
pip install ultralytics==8.3.180 opencv-python
```

### Using pip (Virtual Environment)

```bash
# Create virtual environment
python -m venv yolo-video-env

# Activate (Windows)
yolo-video-env\Scripts\activate

# Activate (macOS/Linux)
source yolo-video-env/bin/activate

# Install packages
pip install ultralytics==8.3.180 opencv-python
```

## 📁 Project Structure

```
yolov8-video-processor/
├── yolo_video_processor.py    # Main application
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── examples/                  # Example videos and models
│   ├── sample_video.mp4
│   └── best.pt
└── outputs/                   # Processed videos
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLOv8 implementation
- OpenCV community for computer vision tools
- Contributors and users of this project
