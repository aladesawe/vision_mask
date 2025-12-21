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
    
    @property
    def avg_detection_ms(self) -> float:
        return (sum(self.detection_times) / len(self.detection_times) * 1000) if self.detection_times else 0
    
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
    """Load YOLO model"""
    model = YOLO('yolov8n.pt')
    return model

def apply_mask(frame, boxes, mask_type='blur', blur_strength=51):
    """Apply mask to detected objects in frame"""
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

def process_video(video_path, model, target_classes, mask_type, blur_strength, confidence_threshold, progress_callback=None, batch_size=8):
    """Process video and mask detected objects with batched inference"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
    
    frame_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if frame_buffer:
                frame_start = time.perf_counter()
                
                detection_start = time.perf_counter()
                results = model(frame_buffer, conf=confidence_threshold, verbose=False)
                detection_end = time.perf_counter()
                batch_detection_time = detection_end - detection_start
                per_frame_detection = batch_detection_time / len(frame_buffer)
                
                for i, (frm, result) in enumerate(zip(frame_buffer, results)):
                    timing_stats.detection_times.append(per_frame_detection)
                    
                    boxes_to_mask = []
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            if class_id in target_classes:
                                boxes_to_mask.append(box.xyxy[0].cpu().numpy())
                    
                    masking_start = time.perf_counter()
                    masked_frame = apply_mask(frm, boxes_to_mask, mask_type, blur_strength)
                    masking_end = time.perf_counter()
                    timing_stats.masking_times.append(masking_end - masking_start)
                    
                    out.write(masked_frame)
                    frame_count += 1
                    
                    if progress_callback and total_frames > 0:
                        progress_callback(frame_count / total_frames)
                
                frame_end = time.perf_counter()
                for _ in range(len(frame_buffer)):
                    timing_stats.frame_times.append((frame_end - frame_start) / len(frame_buffer))
            break
        
        frame_buffer.append(frame)
        
        if len(frame_buffer) >= batch_size:
            frame_start = time.perf_counter()
            
            detection_start = time.perf_counter()
            results = model(frame_buffer, conf=confidence_threshold, verbose=False)
            detection_end = time.perf_counter()
            batch_detection_time = detection_end - detection_start
            per_frame_detection = batch_detection_time / len(frame_buffer)
            
            for i, (frm, result) in enumerate(zip(frame_buffer, results)):
                timing_stats.detection_times.append(per_frame_detection)
                
                boxes_to_mask = []
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        if class_id in target_classes:
                            boxes_to_mask.append(box.xyxy[0].cpu().numpy())
                
                masking_start = time.perf_counter()
                masked_frame = apply_mask(frm, boxes_to_mask, mask_type, blur_strength)
                masking_end = time.perf_counter()
                timing_stats.masking_times.append(masking_end - masking_start)
                
                out.write(masked_frame)
                frame_count += 1
                
                if progress_callback and total_frames > 0:
                    progress_callback(frame_count / total_frames)
            
            frame_end = time.perf_counter()
            for _ in range(len(frame_buffer)):
                timing_stats.frame_times.append((frame_end - frame_start) / len(frame_buffer))
            
            frame_buffer = []
    
    total_end = time.perf_counter()
    timing_stats.total_time = total_end - total_start
    timing_stats.frame_count = frame_count
    
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

def process_image(image, model, target_classes, mask_type, blur_strength, confidence_threshold):
    """Process single image and mask detected objects"""
    timing_stats = TimingStats()
    total_start = time.perf_counter()
    
    detection_start = time.perf_counter()
    results = model(image, conf=confidence_threshold, verbose=False)
    
    boxes_to_mask = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in target_classes:
                    boxes_to_mask.append(box.xyxy[0].cpu().numpy())
    detection_end = time.perf_counter()
    timing_stats.detection_times.append(detection_end - detection_start)
    
    masking_start = time.perf_counter()
    masked_image = apply_mask(image, boxes_to_mask, mask_type, blur_strength)
    masking_end = time.perf_counter()
    timing_stats.masking_times.append(masking_end - masking_start)
    
    total_end = time.perf_counter()
    timing_stats.total_time = total_end - total_start
    timing_stats.frame_count = 1
    timing_stats.frame_times.append(timing_stats.total_time)
    
    return masked_image, len(boxes_to_mask), timing_stats

def main():
    st.title("🎭 Video Object Masking")
    st.markdown("Upload a video or image and mask specific objects using YOLO object detection.")
    
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
            options=['blur', 'pixelate', 'black', 'color'],
            index=0,
            help="Choose how detected objects should be masked"
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
        
        st.subheader("Detection Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Minimum confidence for object detection"
        )
    
    tab1, tab2 = st.tabs(["📹 Video Processing", "🖼️ Image Processing"])
    
    with tab1:
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
            
            if st.button("🎬 Process Video", type="primary", key="process_video"):
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
                        
                        st.subheader("Latency Analysis")
                        st.markdown("*Timing breakdown for real-time pipeline evaluation:*")
                        
                        lat_col1, lat_col2, lat_col3 = st.columns(3)
                        with lat_col1:
                            st.metric("Avg Detection", f"{timing_stats.avg_detection_ms:.2f} ms")
                            st.metric("Avg Masking", f"{timing_stats.avg_masking_ms:.2f} ms")
                        with lat_col2:
                            st.metric("Avg Frame Total", f"{timing_stats.avg_frame_ms:.2f} ms")
                            st.metric("Theoretical FPS", f"{timing_stats.theoretical_fps:.1f}")
                        with lat_col3:
                            st.metric("Min Frame", f"{timing_stats.min_frame_ms:.2f} ms")
                            st.metric("Max Frame", f"{timing_stats.max_frame_ms:.2f} ms")
                        
                        st.info(f"**Total processing time:** {timing_stats.total_time:.2f}s for {frame_count} frames | "
                                f"**Introduced latency per frame:** {timing_stats.avg_frame_ms:.2f} ms")
                        
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
    
    with tab2:
        st.subheader("Image Upload")
        
        uploaded_image = st.file_uploader(
            "Upload an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="image_uploader"
        )
        
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("Could not decode the uploaded image. Please upload a valid image file.")
                st.stop()
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image:**")
                st.image(image_rgb, use_container_width=True)
            
            if st.button("🎨 Process Image", type="primary", key="process_image"):
                if not selected_objects:
                    st.error("Please select at least one object type to mask.")
                else:
                    target_class_ids = [CLASS_TO_ID[obj] for obj in selected_objects if obj in CLASS_TO_ID]
                    
                    try:
                        with st.spinner("Processing image..."):
                            masked_image, detection_count, timing_stats = process_image(
                                image,
                                model,
                                target_class_ids,
                                mask_type,
                                blur_strength,
                                confidence_threshold
                            )
                        
                        masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
                        
                        with col2:
                            st.markdown("**Masked Image:**")
                            st.image(masked_image_rgb, use_container_width=True)
                        
                        st.success(f"Detected and masked {detection_count} object(s).")
                        
                        st.subheader("Latency Analysis")
                        lat_col1, lat_col2, lat_col3 = st.columns(3)
                        with lat_col1:
                            st.metric("Detection Time", f"{timing_stats.avg_detection_ms:.2f} ms")
                        with lat_col2:
                            st.metric("Masking Time", f"{timing_stats.avg_masking_ms:.2f} ms")
                        with lat_col3:
                            st.metric("Total Latency", f"{timing_stats.total_time * 1000:.2f} ms")
                        
                        _, buffer = cv2.imencode('.png', masked_image)
                        st.download_button(
                            label="📥 Download Masked Image",
                            data=buffer.tobytes(),
                            file_name="masked_image.png",
                            mime="image/png"
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    ### How to Use
    1. **Select Objects**: Use the sidebar to choose which objects you want to mask (default: person)
    2. **Choose Mask Type**: Select how you want objects to be masked (blur, pixelate, black, or colored)
    3. **Upload Media**: Upload a video or image file
    4. **Process**: Click the process button to apply masking
    5. **Download**: Download the processed file with masked objects
    
    ### Supported Objects
    This application uses YOLO (You Only Look Once) object detection which can detect 80 different object types including:
    - People, vehicles (cars, trucks, buses, motorcycles, bicycles)
    - Animals (dogs, cats, birds, horses, etc.)
    - Common objects (phones, laptops, bags, bottles, etc.)
    """)

if __name__ == "__main__":
    main()
