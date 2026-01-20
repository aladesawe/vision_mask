# Video Object Masking Application

## Overview
A Streamlit-based application that uses YOLO segmentation to automatically detect and mask objects in videos. Users can upload video files, select which object types to mask, choose from multiple masking options including mannequin replacement, and download the processed output.

## Recent Changes
- 2026-01-20: Added mannequin replacement using YOLO segmentation for precise object boundaries
- 2026-01-20: Switched to YOLOv8n-seg (segmentation model) for precise masking
- 2026-01-20: Removed image processing, focused on video only
- 2026-01-20: Added adaptive frame skipping for faster processing (target: process in video duration, max 60s)
- 2025-12-21: Added H.264 conversion for browser-compatible video playback
- 2025-12-21: Added timing instrumentation for latency analysis
- 2025-12-20: Initial implementation with YOLO object detection and masking

## Project Architecture

### Tech Stack
- **Frontend**: Streamlit
- **Object Detection**: YOLOv8n-seg (ultralytics segmentation model)
- **Video Processing**: OpenCV, FFmpeg
- **Runtime**: Python 3.11

### File Structure
```
/
├── app.py              # Main Streamlit application
├── pyproject.toml      # Python dependencies
├── .streamlit/
│   └── config.toml     # Streamlit server configuration
└── replit.md           # Project documentation
```

### Key Features
1. **Video Processing**: Upload videos and mask specific objects with segmentation precision
2. **Mannequin Replacement**: Replace detected humans with a mannequin silhouette
3. **Configurable Object Selection**: Choose from 80 YOLO object classes
4. **Multiple Mask Types**: Blur, pixelate, black, color, or mannequin masking
5. **Adaptive Frame Skipping**: Smart frame skipping to meet target processing time
6. **Performance Metrics**: Detailed timing analysis for latency evaluation
7. **Browser-Compatible Output**: H.264 encoding for playback in browsers

### Mask Types
- **blur**: Gaussian blur effect (adjustable strength)
- **pixelate**: Mosaic/pixelation effect
- **black**: Solid black overlay
- **color**: Green color overlay
- **mannequin**: Replace with mannequin silhouette (segmentation-based)

### Speed Optimizations
- Adaptive frame skipping based on video duration vs max processing time (60s cap)
- Reuses detection results for skipped frames
- YOLO nano model for faster inference

### Supported Object Types
The application can detect and mask 80 different object types including:
- People
- Vehicles (cars, trucks, buses, motorcycles, bicycles)
- Animals (dogs, cats, birds, horses, etc.)
- Common objects (phones, laptops, bags, bottles, etc.)

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Configuration
- Server binds to 0.0.0.0:5000
- YOLO segmentation model (yolov8n-seg.pt) is downloaded on first run
- FFmpeg required for H.264 video conversion

## Browser access
In your browser, navigate to localhost:5000
