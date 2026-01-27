# Video Object Masking
## Motivation
I came across a court ruling in [Washington State, USA](https://www.king5.com/article/news/investigations/investigators/judge-orders-washington-police-release-surveillance-camera-data-privacy-questions/281-c2037d52-6afb-4bf7-95ad-0eceaf477864), where images captured by automated license plate readers qualify as public records and is subject to the state's Public Record Act. We typically think nothing of these some-what ubiquitous devices while we go about our daily business, but now, with this ruling, there's now significant privacy and safety concern about who, and to what ends, can these captured images be obtained, and used.

Law enforcement uses various means to aid their enforcement duties: in responding, real-time, to crime, monitoring crime hot-spots, and establishing ground-truth of events during investigations. Washington State, like most developed tech-driven regions, deploys a network of automated cameras on highways and streets, and captures vehicles and occupants in its law-enforcement needs. There's an argument to be made of how much surveillance is sufficent for public safety; that threshold will be debatable, and will vary as widely as the survey respondents. Add the recent ruling, that such captures are indeed public records, and one may start to question the cost-benefit. 

What if there's a way to address the needs of:
- Law enforcement agencies, to empower them in their public safety endeavors
- Privacy advocates, who argue that having these identifiable information of private citizens, now adjudged public record, represent a potential safety concern given their daily routines can now easily be reconstructed using these records without them even consenting to it.

One solution is a one-way Homomorphic encryption could work as law enforcement get the ability to perform searches of persons/objects of interest against irreversibly, but computably, encrypted data, given an appropriate confidence level.

Another would be to have the captured data pre-processed and stored with masks on all personally-identifiable information, safe for those under an existing warrant. For instance, the following scenario plays out:
- Law enforcement receives a report of a stolen vehicle, or a suspect on the run, at time T
- All image/video captures just before and until time T, had all license plates, and faces masked when stored
- From time T+1, when images/videos are processed, all those that match, within a specified confidence threshold, against a database of active investigations or warrants are stored without masks
- All other non-matched entities are stored with masks

Masking alternatives are explored [below](#mask-types).

## Results
Using a video captured on a visit to the archeological site, Teotihuacan while visiting Mexico city, we mask people using a variety of the options, for a 0.3 detection confidence threshold.

### Original Video
![View original video - tourists at Teotihuacan archaeological site](https://github.com/aladesawe/vision_mask/blob/e160e83d4bf7f07e0e083c8219a764c9ea4337b8/assets/test_capture.MOV)

## Model Comparison

| Original Frame |  |
|---|---|
![Original Frame](https://github.com/aladesawe/vision_mask/blob/b555dfa762898072273bf890d6d9cf02e0fe7367/assets/original_frame.jpg)
| Gemini 3 Pro Image | Gemini 2.5 Flash Image |
| ![Gemini 3 Result](https://github.com/aladesawe/vision_mask/blob/b555dfa762898072273bf890d6d9cf02e0fe7367/assets/ai_preview_frame_190_gemini-3-pro-image-preview_better.png) | ![Gemini 2.5 Result](https://github.com/aladesawe/vision_mask/blob/b555dfa762898072273bf890d6d9cf02e0fe7367/assets/ai_preview_frame_190_gemini-2-5-flash-image.png) |

### Observation
When testing out different Gemini models, I got varying results with the pro-image version 3 giving the most precise in-place replacement of detected people with mannequins, more often than not.
With Gemini 2.5 Image model, there are misplaced mannequin, top left on the pyramid's steps.

## Other Masking types
### Pixelated Masking
![Pixelated masking](https://github.com/aladesawe/vision_mask/blob/b555dfa762898072273bf890d6d9cf02e0fe7367/assets/pixelated_video.mp4)

### Color Masking
![Masking with color](https://github.com/aladesawe/vision_mask/blob/b555dfa762898072273bf890d6d9cf02e0fe7367/assets/color_mask_video.mp4)

### Blurring, of strength 91
![Blurring of strength 91](https://github.com/aladesawe/vision_mask/blob/b555dfa762898072273bf890d6d9cf02e0fe7367/assets/blurred_video.mp4)

### Black Masking
![Masking with color black](https://github.com/aladesawe/vision_mask/blob/b555dfa762898072273bf890d6d9cf02e0fe7367/assets/black_color_mask_video.mp4)

### Mannequin Silhouette
![A Mannequin silhoutte](https://github.com/aladesawe/vision_mask/blob/b555dfa762898072273bf890d6d9cf02e0fe7367/assets/mannequin_video.mp4)

## Improvements
In real-world applications, there may be need for annotations to indicate objects exempted from masking. There's a possibility for this to be handled with a bit of prompt engineering as well.

A significant improvement to the masking approach is to generate the cryptographic hash of the detected object's data (class label, bounding box, and feature embeddings of the object) and storing only the hash; bypassing the need to store the images or even generate masks. These digial signatures can then be used for law enforcement purposes without the privacy risk associated with capturing the raw footage.


## Other Applications
- Masking patients in hospitals/operating rooms where video recordings are captured for training purposes
- Selectively masking sections of an audience who do not consent to a particular broadcast channel


## Implementation
A Streamlit-based application that uses YOLO segmentation to automatically detect and mask objects in videos. Users can upload video files, select which object types to mask, choose from multiple masking options including mannequin silhouette replacement, and download the processed output. We also add an AI-mannequin replacement to preview genAI image editing. The type of the mask, and whether frame skipping is enabled during processing, impacts the quality of the result and speed of processing.

## Features
- **Video Processing**: Upload videos and mask specific objects with segmentation precision
- **Mannequin Silhouette**: Replace detected humans with a mannequin silhouette
- **AI Preview Mannequin (Gemini)**: Single-frame AI-powered replacement using Google Gemini
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
### 0. Clone the repository, and setup your virtual environment

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

## Speed Optimizations
- Adaptive frame skipping based on video duration vs max processing time (60s cap), if enabled
- Reuses detection results for skipped frames (except mannequin mode)
- Note: Frame skipping is disabled for mannequin mode to ensure precise alignment between the mask and person position

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
