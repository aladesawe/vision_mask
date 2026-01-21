import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import base64
import io

# Try to import google-genai for Gemini, but make it optional
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None

YOLO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

CLASS_TO_ID = {v: k for k, v in YOLO_CLASSES.items()}

@dataclass
class TimingStats:
    """Container for timing statistics"""
    detection_times: List[float] = field(default_factory=list)
    masking_times: List[float] = field(default_factory=list)
    frame_times: List[float] = field(default_factory=list)
    total_time: float = 0.0
    frame_count: int = 0
    frame_skip: int = 1
    detections_run: int = 0
    
    @property
    def avg_detection_ms(self) -> float:
        active_detections = [t for t in self.detection_times if t > 0]
        return (sum(active_detections) / len(active_detections) * 1000) if active_detections else 0
    
    @property
    def avg_masking_ms(self) -> float:
        return (sum(self.masking_times) / len(self.masking_times) * 1000) if self.masking_times else 0
    
    @property
    def avg_frame_ms(self) -> float:
        return (sum(self.frame_times) / len(self.frame_times) * 1000) if self.frame_times else 0
    
    @property
    def min_frame_ms(self) -> float:
        return min(self.frame_times) * 1000 if self.frame_times else 0
    
    @property
    def max_frame_ms(self) -> float:
        return max(self.frame_times) * 1000 if self.frame_times else 0
    
    @property
    def theoretical_fps(self) -> float:
        return 1000 / self.avg_frame_ms if self.avg_frame_ms > 0 else 0

st.set_page_config(
    page_title="Video Object Masking",
    page_icon="🎭",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load YOLO segmentation model for precise masking"""
    model = YOLO('yolov8n-seg.pt')
    return model

def apply_mask(frame, boxes, mask_type='blur', blur_strength=51):
    """Apply mask to detected objects in frame (bounding box based)"""
    result_frame = frame.copy()
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        roi = result_frame[y1:y2, x1:x2]
        
        if mask_type == 'blur':
            blur_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            masked_roi = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
        elif mask_type == 'pixelate':
            h, w = roi.shape[:2]
            if h > 0 and w > 0:
                temp = cv2.resize(roi, (max(1, w // 10), max(1, h // 10)), interpolation=cv2.INTER_LINEAR)
                masked_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                masked_roi = roi
        elif mask_type == 'black':
            masked_roi = np.zeros_like(roi)
        elif mask_type == 'color':
            masked_roi = np.full_like(roi, (0, 255, 0))
        else:
            masked_roi = roi
            
        result_frame[y1:y2, x1:x2] = masked_roi
    
    return result_frame

def apply_segmentation_mask(frame, result, target_classes, mask_type='blur', blur_strength=51):
    """Apply mask using segmentation for precise object boundaries"""
    result_frame = frame.copy()
    
    if result.masks is None or result.boxes is None:
        return result_frame
    
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes
    
    combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])
        if class_id not in target_classes:
            continue
        
        if i >= len(masks):
            continue
            
        mask = masks[i]
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        binary_mask = (mask_resized > 0.5).astype(np.uint8)
        combined_mask = np.maximum(combined_mask, binary_mask)
    
    if combined_mask.sum() == 0:
        return result_frame
    
    mask_3ch = np.stack([combined_mask] * 3, axis=-1)
    
    if mask_type == 'mannequin':
        mannequin_color = np.full_like(frame, (180, 180, 180))
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_mask = np.zeros_like(combined_mask)
        cv2.drawContours(edge_mask, contours, -1, (1,), 3)
        edge_3ch = np.stack([edge_mask] * 3, axis=-1)
        
        result_frame = np.where(mask_3ch, mannequin_color, result_frame)
        
        edge_color = np.full_like(frame, (120, 120, 120))
        result_frame = np.where(edge_3ch, edge_color, result_frame)
        
    elif mask_type == 'blur':
        blur_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
        result_frame = np.where(mask_3ch, blurred, result_frame)
        
    elif mask_type == 'pixelate':
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // 20), max(1, h // 20)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        result_frame = np.where(mask_3ch, pixelated, result_frame)
        
    elif mask_type == 'black':
        result_frame = np.where(mask_3ch, 0, result_frame)
        
    elif mask_type == 'color':
        green = np.full_like(frame, (0, 255, 0))
        result_frame = np.where(mask_3ch, green, result_frame)
    
    return result_frame

def convert_to_h264(input_path, output_path):
    """Convert video to H.264 codec for browser compatibility"""
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            output_path
        ], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def process_video(video_path, model, target_classes, mask_type, blur_strength, confidence_threshold, progress_callback=None, max_processing_time=60.0):
    """Process video with segmentation-based masking and adaptive frame skipping.
    
    Uses YOLO segmentation for precise object boundaries and supports mannequin replacement.
    Note: Frame skipping is disabled for mannequin mode to ensure precise alignment.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_duration = total_frames / fps if fps > 0 else 0
    target_time = min(video_duration, max_processing_time)
    
    # Disable frame skipping for mannequin mode - precise alignment is critical
    if mask_type == 'mannequin':
        frame_skip = 1  # Process every frame
    else:
        estimated_time_per_detection = 0.2
        max_detections = int(target_time / estimated_time_per_detection) if estimated_time_per_detection > 0 else total_frames
        max_detections = max(1, max_detections)
        frame_skip = max(1, total_frames // max_detections)
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        temp_output_path = tmp_file.name
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise ValueError("Could not create output video file")
    
    frame_count = 0
    timing_stats = TimingStats()
    total_start = time.perf_counter()
    
    current_result = None
    frames_since_detection = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.perf_counter()
        
        run_detection = (frames_since_detection >= frame_skip) or (frame_count == 0)
        
        if run_detection:
            detection_start = time.perf_counter()
            
            results = model(frame, conf=confidence_threshold, verbose=False)
            current_result = results[0] if results else None
            
            detection_end = time.perf_counter()
            timing_stats.detection_times.append(detection_end - detection_start)
            frames_since_detection = 0
        else:
            timing_stats.detection_times.append(0)
            frames_since_detection += 1
        
        masking_start = time.perf_counter()
        
        if current_result is not None:
            masked_frame = apply_segmentation_mask(
                frame, current_result, target_classes, 
                mask_type, blur_strength
            )
        else:
            masked_frame = frame.copy()
            
        masking_end = time.perf_counter()
        timing_stats.masking_times.append(masking_end - masking_start)
        
        out.write(masked_frame)
        
        frame_end = time.perf_counter()
        timing_stats.frame_times.append(frame_end - frame_start)
        
        frame_count += 1
        if progress_callback and total_frames > 0:
            progress_callback(frame_count / total_frames)
    
    total_end = time.perf_counter()
    timing_stats.total_time = total_end - total_start
    timing_stats.frame_count = frame_count
    timing_stats.frame_skip = frame_skip
    timing_stats.detections_run = len([t for t in timing_stats.detection_times if t > 0])
    
    cap.release()
    out.release()
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as final_file:
        final_output_path = final_file.name
    
    if convert_to_h264(temp_output_path, final_output_path):
        os.unlink(temp_output_path)
        return final_output_path, frame_count, timing_stats
    else:
        os.unlink(final_output_path)
        return temp_output_path, frame_count, timing_stats

def find_frame_with_most_people(video_path, model, confidence_threshold=0.5, sample_rate=10):
    """Analyze video to find frame with most people detected.
    
    Returns: (frame_index, frame, person_count, result)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, 0, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best_frame_idx = 0
    best_frame = None
    best_count = 0
    best_result = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            results = model(frame, conf=confidence_threshold, verbose=False)
            if results and results[0].boxes is not None:
                person_count = sum(1 for cls in results[0].boxes.cls if int(cls) == 0)
                if person_count > best_count:
                    best_count = person_count
                    best_frame_idx = frame_idx
                    best_frame = frame.copy()
                    best_result = results[0]
        
        frame_idx += 1
    
    cap.release()
    return best_frame_idx, best_frame, best_count, best_result


def create_mask_image_for_ai(frame, result, target_class_id=0):
    """Create a mask image for OpenAI image editing.
    
    Creates a PNG with alpha channel where masked areas are transparent.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if result.masks is None or result.boxes is None:
        return None
    
    for i, (seg_mask, box) in enumerate(zip(result.masks.data, result.boxes)):
        cls_id = int(box.cls[0])
        if cls_id == target_class_id:
            seg_mask_np = seg_mask.cpu().numpy()
            seg_mask_resized = cv2.resize(seg_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, (seg_mask_resized > 0.5).astype(np.uint8) * 255)
    
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask


def ai_replace_people_with_mannequins(frame, mask):
    """Use Gemini image generation API to replace people with mannequins.
    
    Args:
        frame: Original BGR frame from OpenCV
        mask: Binary mask where white (255) indicates areas to replace
    
    Returns:
        Edited frame as numpy array (BGR), or None if failed
    """
    try:
        from PIL import Image
        
        client = genai.Client(
            api_key=os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY"),
            http_options={
                "api_version": "",
                "base_url": os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")
            }
        )
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w = frame.shape[:2]
        max_size = 1024
        scale = min(max_size / w, max_size / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h))
        
        img_pil = Image.fromarray(frame_resized, 'RGB')
        
        prompt = (
            "Edit this image: Replace all people (the masked/highlighted areas in white on the mask) "
            "with realistic gray mannequins or dress forms. "
            "The mannequins should be solid gray, featureless human-shaped figures that naturally fit the scene. "
            "Keep the background and all other elements exactly as they are. "
            "The mannequins should have the same pose and position as the original people."
        )
        
        mask_pil = Image.fromarray(mask_resized, 'L')
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt, img_pil],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )
        
        if response.candidates and len(response.candidates) > 0:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    img_bytes = part.inline_data.data
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    result_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if result_img is not None:
                        result_resized = cv2.resize(result_img, (w, h))
                        return result_resized
        
        return None
        
    except Exception as e:
        st.error(f"AI image editing failed: {str(e)}")
        return None


def main():
    st.title("🎭 Video Object Masking")
    st.markdown("Upload a video and mask specific objects using YOLO object detection.")
    
    with st.spinner("Loading YOLO model..."):
        model = load_model()
    
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Object Selection")
        common_objects = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 
                          'dog', 'cat', 'bird', 'cell phone', 'laptop']
        
        default_selection = ['person']
        
        selected_objects = st.multiselect(
            "Select objects to mask:",
            options=sorted(YOLO_CLASSES.values()),
            default=default_selection,
            help="Choose which objects should be masked in the video"
        )
        
        st.subheader("Mask Settings")
        
        mask_type = st.selectbox(
            "Mask Type:",
            options=['blur', 'pixelate', 'black', 'color', 'mannequin'],
            index=0,
            help="Choose how detected objects should be masked. 'mannequin' replaces humans with a silhouette."
        )
        
        if mask_type == 'blur':
            blur_strength = st.slider(
                "Blur Strength:",
                min_value=11,
                max_value=101,
                value=51,
                step=10,
                help="Higher values create more blur"
            )
        else:
            blur_strength = 51
        
        if mask_type == 'mannequin':
            st.info("Mannequin mode uses segmentation to replace detected people with a mannequin silhouette.")
        
        st.divider()
        st.subheader("AI Replacement (Preview)")
        st.markdown("*Try AI-powered replacement on a single frame*")
        if not GEMINI_AVAILABLE:
            st.info("Gemini package not installed. AI preview is unavailable.")
            use_ai_preview = False
        else:
            use_ai_preview = st.checkbox(
                "Enable AI Frame Preview",
                value=False,
                help="Finds the frame with most people and uses AI to replace them with mannequins"
            )
            if use_ai_preview:
                st.warning("AI preview uses Gemini's image generation API. Cost: ~$0.04 per frame (billed to Replit credits).")
        
        st.divider()
        st.subheader("Detection Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Minimum confidence for object detection"
        )
    
    st.subheader("Video Upload")
    
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="video_uploader"
    )
    
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(uploaded_video.read())
            input_video_path = tmp_file.name
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Video:**")
            st.video(input_video_path)
        
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            process_clicked = st.button("🎬 Process Video", type="primary", key="process_video")
        
        with btn_col2:
            if use_ai_preview:
                ai_preview_clicked = st.button("🤖 AI Preview (Single Frame)", key="ai_preview")
            else:
                ai_preview_clicked = False
        
        if ai_preview_clicked and use_ai_preview:
            with st.spinner("Finding frame with most people..."):
                frame_idx, best_frame, person_count, best_result = find_frame_with_most_people(
                    input_video_path, model, confidence_threshold, sample_rate=5
                )
            
            if best_frame is not None and person_count > 0:
                st.success(f"Found frame {frame_idx} with {person_count} people detected.")
                
                ai_col1, ai_col2 = st.columns(2)
                
                with ai_col1:
                    st.markdown("**Original Frame:**")
                    st.image(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                mask = create_mask_image_for_ai(best_frame, best_result, target_class_id=0)
                
                if mask is not None:
                    with st.spinner("Calling AI to replace people with mannequins..."):
                        ai_result = ai_replace_people_with_mannequins(best_frame, mask)
                    
                    with ai_col2:
                        if ai_result is not None:
                            st.markdown("**AI Mannequin Replacement:**")
                            st.image(cv2.cvtColor(ai_result, cv2.COLOR_BGR2RGB), use_container_width=True)
                        else:
                            st.markdown("**Segmentation Mask (for reference):**")
                            st.image(mask, use_container_width=True)
                            st.warning("AI replacement failed. Showing the mask that would be used.")
                else:
                    st.warning("Could not create mask for AI editing.")
            else:
                st.warning("No people detected in the video.")
        
        if process_clicked:
            if not selected_objects:
                st.error("Please select at least one object type to mask.")
            else:
                target_class_ids = [CLASS_TO_ID[obj] for obj in selected_objects if obj in CLASS_TO_ID]
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {int(progress * 100)}%")
                
                try:
                    with st.spinner("Processing video..."):
                        output_path, frame_count, timing_stats = process_video(
                            input_video_path,
                            model,
                            target_class_ids,
                            mask_type,
                            blur_strength,
                            confidence_threshold,
                            update_progress
                        )
                    
                    progress_bar.progress(1.0)
                    status_text.text(f"Completed! Processed {frame_count} frames.")
                    
                    st.subheader("Performance Analysis")
                    st.markdown("*Timing breakdown and optimization metrics:*")
                    
                    lat_col1, lat_col2, lat_col3, lat_col4 = st.columns(4)
                    with lat_col1:
                        st.metric("Avg Detection", f"{timing_stats.avg_detection_ms:.2f} ms")
                        st.metric("Avg Masking", f"{timing_stats.avg_masking_ms:.2f} ms")
                    with lat_col2:
                        st.metric("Avg Frame Total", f"{timing_stats.avg_frame_ms:.2f} ms")
                        st.metric("Theoretical FPS", f"{timing_stats.theoretical_fps:.1f}")
                    with lat_col3:
                        st.metric("Frame Skip", f"1:{timing_stats.frame_skip}")
                        st.metric("Detections Run", f"{timing_stats.detections_run}")
                    with lat_col4:
                        st.metric("Total Frames", f"{frame_count}")
                        speedup = frame_count / timing_stats.detections_run if timing_stats.detections_run > 0 else 1
                        st.metric("Speedup Factor", f"{speedup:.1f}x")
                    
                    st.info(f"**Total processing time:** {timing_stats.total_time:.2f}s for {frame_count} frames | "
                            f"**Detection ran on {timing_stats.detections_run} frames** (skipped {frame_count - timing_stats.detections_run})")
                    
                    with col2:
                        st.markdown("**Masked Video:**")
                        st.video(output_path)
                    
                    with open(output_path, 'rb') as f:
                        video_data = f.read()
                    
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                    
                    st.download_button(
                        label="📥 Download Masked Video",
                        data=video_data,
                        file_name="masked_video.mp4",
                        mime="video/mp4"
                    )
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                finally:
                    if os.path.exists(input_video_path):
                        os.unlink(input_video_path)
    
    st.markdown("---")
    st.markdown("""
    ### How to Use
    1. **Select Objects**: Use the sidebar to choose which objects you want to mask (default: person)
    2. **Choose Mask Type**: Select how you want objects to be masked:
       - **blur**: Gaussian blur effect
       - **pixelate**: Mosaic/pixelation effect  
       - **black**: Solid black overlay
       - **color**: Green color overlay
       - **mannequin**: Replace humans with a mannequin silhouette (uses segmentation)
    3. **Upload Video**: Upload a video file (MP4, AVI, MOV, MKV)
    4. **Process**: Click the process button to apply masking
    5. **Download**: Download the processed video with masked objects
    
    ### Supported Objects
    This application uses YOLO segmentation which can detect and precisely mask 80 different object types including:
    - People, vehicles (cars, trucks, buses, motorcycles, bicycles)
    - Animals (dogs, cats, birds, horses, etc.)
    - Common objects (phones, laptops, bags, bottles, etc.)
    """)

if __name__ == "__main__":
    main()
