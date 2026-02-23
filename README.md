# Video Object Masking
## Motivation
I came across a court ruling in [Washington State, USA](https://www.king5.com/article/news/investigations/investigators/judge-orders-washington-police-release-surveillance-camera-data-privacy-questions/281-c2037d52-6afb-4bf7-95ad-0eceaf477864), which held that images captured by automated license plate readers qualify as public records and are therefore subject to the state’s Public Records Act. We typically think little of these somewhat ubiquitous devices as we go about our daily lives. However, following this ruling, significant privacy and safety concerns now arise around who can access these images, under what conditions, and for what purposes they may be obtained and used.

Law enforcement agencies use a variety of tools to support their public safety responsibilities: responding in real time to incidents, monitoring crime hotspots, and establishing ground truth during investigations. Washington State, like most technologically advanced regions, deploys a network of automated cameras on highways and streets, capturing vehicles-and sometimes occupants-in service of these objectives. There is an ongoing debate about how much surveillance is sufficient for public safety; this threshold is subjective and varies widely depending on perspective. When combined with the recent ruling that such data constitutes public record, the cost–benefit tradeoff becomes even more complex.

What if there were a way to address the needs of:

- Law enforcement agencies, empowering them to effectively pursue public safety objectives.

- Privacy advocates, who argue that making identifiable information of private citizens publicly accessible introduces substantial privacy and safety risks, particularly when individuals’ daily routines can be reconstructed without their knowledge or consent.

One potential approach is to apply one-way homomorphic encryption, enabling law enforcement to perform searches for persons or objects of interest over irreversibly—but computably—encrypted data, subject to appropriate confidence thresholds.

Another approach would involve pre-processing captured data and storing it with all personally identifiable information (PII) masked by default, except for cases supported by an existing warrant. For example:

- Law enforcement receives a report of a stolen vehicle or a suspect at large at time T.

- All image and video captures prior to and up to time T are stored with license plates and faces masked.

- From time T+1 onward, newly captured frames are compared against databases associated with active investigations or warrants, and only those matching above a defined confidence threshold are stored unmasked.

- All non-matching captures remain masked.

Masking alternatives are explored [below](#mask-types).

## Results
Using a video recorded during a visit to the archaeological site of Teotihuacan near Mexico City, we applied several masking techniques to anonymize individuals, using a detection confidence threshold of 0.3.

### Original Video
<video src="https://raw.githubusercontent.com/aladesawe/vision_mask/e160e83d4bf7f07e0e083c8219a764c9ea4337b8/assets/test_capture.MOV" type="video/quicktime; codecs=h264" aria-label="View original video - tourists at Teotihuacan archaeological site" width="320" height="240" controls></video>

## Model Comparison

| Original Frame |  |
|---|---|
![Original Frame](https://raw.githubusercontent.com/aladesawe/vision_mask/b555dfa762898072273bf890d6d9cf02e0fe7367/assets/original_frame.jpg)
| Gemini 3 Pro Image | Gemini 2.5 Flash Image |
| ![Gemini 3 Result](https://raw.githubusercontent.com/aladesawe/vision_mask/b555dfa762898072273bf890d6d9cf02e0fe7367/assets/ai_preview_frame_190_gemini-3-pro-image-preview_better.png) | ![Gemini 2.5 Result](https://raw.githubusercontent.com/aladesawe/vision_mask/b555dfa762898072273bf890d6d9cf02e0fe7367/assets/ai_preview_frame_190_gemini-2-5-flash-image.png) |

### Observation
When evaluating different Gemini models, we observed varying performance. The Gemini 3 Pro Image model produced the most precise in-place replacement of detected individuals with mannequin silhouettes in the majority of cases. In contrast, Gemini 2.5 Image occasionally introduced misplaced mannequins—for example, incorrectly inserting figures on the pyramid steps—and often failed to replace many, if not all, individuals present in the frame.

## Other Masking types
### Pixelated Masking
<video src="https://raw.githubusercontent.com/aladesawe/vision_mask/64081dfb23f9be2c059a2f8227dcba0c5a3afca1/assets/pixelated_video.mp4" type="video/mp4; codecs=h264" aria-label="Pixelated masking" width="320" height="240" controls></video>

### Color Masking
<video src="https://raw.githubusercontent.com/aladesawe/vision_mask/64081dfb23f9be2c059a2f8227dcba0c5a3afca1/assets/color_mask_video.mp4" type="video/mp4; codecs=h264" aria-label="Masking with green" width="320" height="240" controls></video>

### Blurring, of strength 91
<video src="https://raw.githubusercontent.com/aladesawe/vision_mask/64081dfb23f9be2c059a2f8227dcba0c5a3afca1/assets/blurred_video.mp4" type="video/mp4; codecs=h264" aria-label="Blurring with mask strength 91" width="320" height="240" controls></video>

### Black Masking
<video src="https://raw.githubusercontent.com/aladesawe/vision_mask/64081dfb23f9be2c059a2f8227dcba0c5a3afca1/assets/black_color_mask_video.mp4" type="video/mp4; codecs=h264" aria-label="Masking with black" width="320" height="240" controls></video>

### Mannequin Silhouette
<video src="https://raw.githubusercontent.com/aladesawe/vision_mask/64081dfb23f9be2c059a2f8227dcba0c5a3afca1/assets/mannequin_video.mp4" type="video/mp4; codecs=h264" aria-label="Mannequin Silhoutte" width="320" height="240" controls></video>

## Improvements
In real-world deployments, there may be a need for explicit annotations to designate objects or individuals exempt from masking. This could potentially be addressed through prompt engineering or policy-driven metadata injection.

A more significant enhancement would be to compute cryptographic hashes of detected object metadata-including class labels, bounding boxes, and feature embeddings-and store only these hashes rather than the raw images or masked frames. These digital signatures could support law enforcement search and matching operations while significantly reducing the privacy risks associated with retaining raw surveillance footage.


## Other Applications
- Masking patients in hospitals or operating rooms when video recordings are captured for training or quality assurance.

- Selectively masking members of an audience who have not consented to a particular broadcast or recording.


## Implementation
We implemented a Streamlit-based application that leverages YOLO segmentation to automatically detect and mask objects in video streams. Users can upload video files, select object categories to anonymize, choose from multiple masking strategies-including mannequin silhouette replacement-and download the processed output. We also include a generative AI–based mannequin replacement mode to preview advanced image-editing techniques. The choice of masking method and the use of frame skipping significantly affect both processing speed and output quality.

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
