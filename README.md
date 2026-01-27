# Video Object Masking Application
## Motivation
I came across a court ruling in [Washington State, USA](https://www.king5.com/article/news/investigations/investigators/judge-orders-washington-police-release-surveillance-camera-data-privacy-questions/281-c2037d52-6afb-4bf7-95ad-0eceaf477864), where images captured by automated license plate readers qualify as public records and is subject to the state's Public Record Act. We typically think nothing of these some-what ubiquitous devices while we go about our daily business, but now, with this ruling, there's now significant privacy and safety concern about who, and to what ends, can these captured images be obtained, and used.

Law enforcement uses various means to aid their enforcement duties: in responding real-time to crime, monitoring crime hot-spots, and establishing ground-truth of events during investigations. Washington State, like most developed tech-driven regions, deploys a network of automated cameras on highways and streets, and captures vehicles and occupants in its law-enforcement needs. There's an argument to be made of how much surveillance is sufficent for public safety, and that threshold will be debatable, and vary as widely as the survey respondent. Add the recent ruling, that such captures are indeed public records, and one may start to question the cost-benefit. 

What if there's a way to address the needs of both:
- Law enforcement agencies, to empower them on keeping the public safe
- Public who argue that having these massive collection of private citizens, now adjudged public data, represent a potential safety concern given their daily routines can easily be public accessible without them even consenting to it

Some type of One-way Homomorphic encryption could work as law enforcement get the ability to perform searches of wanted felons/stolen license plates against irreversibly-encrypted data, given an appropriate confidence level.

Another would be to have the data captured pre-processed and stored with masks on all personally-identifiable information, safe for those under an existing warrant. For instance, the following scenario plays out:
- Law enforcement receives a report of a stolen vehicle, or a suspect on the run, at time T
- All image/video captures just before time T, T-1 for instance, had all license plates, and faces masked when stored
- Now from T+1, when images/videos are processed, all those that match, within a specified confidence interval, against a database of active investigations are stored without masks
- All others are stored with masks

Masking alternatives are explored below.

## Overview
A Streamlit-based application that uses YOLO segmentation to automatically detect and mask objects in videos. Users can upload video files, select which object types to mask, choose from multiple masking options including mannequin replacement, and download the processed output. We also add an AI-mannequin replacement to preview genAI image editing. The type of the mask, and whether frame skipping is enabled during processing, impacts the quality and speed of processing.

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


## Results
Using a video captured on a visit to the archeological site, Teotihuacan while visiting Mexico city, we mask people using a variety of the options, for a 0.3 detection confidence threshold.

### Original Video
[View original video - tourists at Teotihuacan archaeological site](assets/test_capture.mov)

### AI Mannequin
[Original Frame](assets/original_frame.jpg)
[AI Mannequin Replacement](assets/gemini-3-pro-image-preview_1.jpg)

### Pixelated Masking
[Pixelated masking](assets/pixelated_video.mp4)

### Color Masking
[Masking with color](assets/color_mask_video.mp4)

### Blurring, of strength 91
[Blurring of strength 91](assets/blurred_video.mp4)

### Black Masking
[Masking with color black](assets/black_color_mask_video.mp4)

### Mannequin Silhouette
[A Mannequin silhoutte](assets/mannequin_video.mp4)