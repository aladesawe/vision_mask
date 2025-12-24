# Video Object Masking Application

## Overview
A Streamlit-based application that uses YOLO (You Only Look Once) object detection to automatically detect and mask objects in videos and images. Users can upload media files, select which object types to mask, and download the processed output.

## Recent Changes
- 2025-12-20: Initial implementation with YOLO object detection and masking

## Project Architecture

### Tech Stack
- **Frontend**: Streamlit
- **Object Detection**: YOLOv8 (ultralytics)
- **Video Processing**: OpenCV
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
1. **Video Processing**: Upload videos and mask specific objects
3. **Configurable Object Selection**: Choose from 80 YOLO object classes
4. **Multiple Mask Types**: Blur, pixelate, black, or color masking
5. **Adjustable Settings**: Confidence threshold and blur strength

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
- YOLO model (yolov8n.pt) is downloaded on first run

## Browser access
In your browser, navigate to localhost:5000
