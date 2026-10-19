# YOLOv8 Animal Tracker

A command-line video processor that runs a custom-trained YOLOv8 model over a video file and writes out an annotated copy, with live progress, speed and ETA reporting. Originally trained to detect animals (the included `best-1.pt` weights), but works with any YOLOv8 `.pt` model.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-video%20I%2FO-5C3EE8?logo=opencv&logoColor=white)

## Features

- Works with any custom-trained YOLOv8 `.pt` model
- Live per-frame progress: percentage, fps, inference time, detection count, elapsed time and ETA
- Two processing modes: frame-by-frame (`manual`, full control and stats) or built-in batch tracking (`batch`, faster)
- End-of-run summary: total time, average fps, average inference time, and a real-time factor (was processing faster or slower than the video's own playback speed?)

## Requirements

- Python 3.8+
- `ultralytics` (YOLOv8)
- `opencv-python`

```bash
pip install ultralytics opencv-python
```

## Usage

```bash
# Frame-by-frame, with detailed live stats
python main.py input_video.mp4 output_video.mp4 --model best-1.pt --conf 0.5 --method manual

# Built-in batch tracking (faster, less detail)
python main.py input_video.mp4 output_video.mp4 --model best-1.pt --method batch
```

| Argument | Description | Default |
|---|---|---|
| `input` | Path to the input video | required |
| `output` | Path to write the annotated video to | required |
| `--model` | Path to a YOLOv8 `.pt` model | `yolov8n.pt` |
| `--conf` | Confidence threshold (0.0–1.0) | `0.25` |
| `--method` | `manual` (frame-by-frame) or `batch` (built-in `model.track`) | `manual` |

## Included model

`best-1.pt` is a YOLOv8 model trained on animal footage and ships in this repo so the script works out of the box. Swap in your own model with `--model`.

## Sample output

```
Loading YOLO model: best-1.pt
Model loaded in 2.45 seconds
Video properties: 1920x1080 @ 30fps, 1500 frames
Estimated video duration: 50.0 seconds
----------------------------------------------------------------------
Frame   1234/1500 ( 82.3%) | Speed:  25.4 fps | Inference:  15.2ms | Detections:  3 | Elapsed: 01:28 | ETA: 00:19
======================================================================
PROCESSING COMPLETE!
======================================================================
Output saved to: output_video.mp4
Frames processed: 1500/1500
Total time: 01:47
Average processing speed: 23.8 fps
Real-time factor: 1.2x
✓ Processing faster than real-time!
```
