# Video Object Masking Application

## Overview
A Streamlit-based application that uses YOLO segmentation to automatically detect and mask objects in videos. Users can upload video files, select which object types to mask, choose from multiple masking options including mannequin replacement, and download the processed output.

## Features
- **Video Processing**: Upload videos and mask specific objects with segmentation precision
- **Mannequin Replacement**: Replace detected humans with a mannequin silhouette
- **AI Preview (Gemini)**: Single-frame AI-powered replacement using Google Gemini
- **Configurable Object Selection**: Choose from 80 YOLO object classes
- **Multiple Mask Types**: Blur, pixelate, black, color, or mannequin masking
- **Adaptive Frame Skipping**: Smart frame skipping to meet target processing time
- **Browser-Compatible Output**: H.264 encoding for playback in browsers

## Tech Stack
- **Frontend**: Streamlit
- **Object Detection**: YOLOv8n-seg (ultralytics segmentation model)
- **Video Processing**: OpenCV, FFmpeg
- **AI Integration**: Google Gemini (for AI preview feature)
- **Runtime**: Python 3.11+

## Running on Replit
The app runs automatically. Just click Run or access via the webview.

## Running Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# or using uv:
uv sync
```

### 2. Set Up Environment Variables
Copy the example environment file and add your API keys:
```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key (required for AI Preview feature):
```
AI_INTEGRATIONS_GEMINI_API_KEY=your_gemini_api_key_here
AI_INTEGRATIONS_GEMINI_BASE_URL=https://generativelanguage.googleapis.com
```

Get a free Gemini API key from: https://aistudio.google.com/apikey

### 3. Run the Application
```bash
streamlit run app.py --server.port 5000
```

### 4. Access in Browser
Navigate to: http://localhost:5000

## Mask Types
- **blur**: Gaussian blur effect (adjustable strength)
- **pixelate**: Mosaic/pixelation effect
- **black**: Solid black overlay
- **color**: Green color overlay
- **mannequin**: Replace with mannequin silhouette (segmentation-based)

## Supported Object Types
The application can detect and mask 80 different object types including:
- People
- Vehicles (cars, trucks, buses, motorcycles, bicycles)
- Animals (dogs, cats, birds, horses, etc.)
- Common objects (phones, laptops, bags, bottles, etc.)

## File Structure
```
/
├── app.py              # Main Streamlit application
├── pyproject.toml      # Python dependencies
├── .env.example        # Environment variables template
├── .streamlit/
│   └── config.toml     # Streamlit server configuration
└── README.md           # This file
```

## Configuration Notes
- YOLO segmentation model (yolov8n-seg.pt) is downloaded on first run
- FFmpeg is required for H.264 video conversion
- Frame skipping is disabled for mannequin mode to ensure precise alignment
