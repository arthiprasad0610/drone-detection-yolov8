"""
live_server_advanced.py
=======================
Advanced Live Video Streaming Server with Object Detection & Smart Capture

PURPOSE:
    Receives frames from an external video processing program.
    Detects flying objects in real-time using YOLOv8.
    Provides UI buttons to capture frames when objects detected or for background training.

API ENDPOINTS:
    GET  /                          - HTML dashboard with capture controls
    POST /submit-frame              - Accept frame from external program
    POST /capture-start             - Start continuous frame capture
    POST /capture-stop              - Stop continuous frame capture
    POST /capture-background        - Capture single background frame
    GET  /live-stream               - MJPEG stream of current feed
    WS   /ws-stream                 - WebSocket for real-time frame updates

USAGE:
    1. Start this server: python live_server_advanced.py
    2. External program sends frames to: http://localhost:7000/submit-frame
    3. Use dashboard buttons to capture frames

EXAMPLE (sending frame from external program):
    import cv2
    import requests
    
    cap = cv2.VideoCapture('video.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Send frame to server
        _, buffer = cv2.imencode('.jpg', frame)
        files = {'frame': buffer.tobytes()}
        response = requests.post('http://localhost:7000/submit-frame', files={'frame': buffer.tobytes()})
        
        if response.status_code == 200:
            print("Frame sent successfully")
"""

# IMPORTANT: torch/YOLO must be imported before cv2/numpy on Windows
# to avoid an OpenMP DLL conflict ([WinError 1114]).
try:
    import torch
    _original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        try:
            return _original_load(*args, **kwargs)
        except Exception as e:
            if 'was not an allowed global' in str(e):
                with torch.serialization.safe_globals([]):
                    try:
                        return _original_load(*args, **{k: v for k, v in kwargs.items() if k != 'weights_only'})
                    except:
                        return _original_load(*args, **kwargs, weights_only=False)
            raise
    torch.load = patched_load
except Exception:
    pass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from sahi.predict import get_sliced_prediction
    from sahi.models.ultralytics import UltralyticsDetectionModel
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

import cv2
import json
import asyncio
import time
import io
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================
# GLOBAL STATE & CONFIGURATION
# ==============================================================

# Video processing paths
VIDEO_DATASET_DIR = Path("input_dataset")
VIDEO_DATASET_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

# State for video processing
video_processing_active = False
video_processing_thread = None

# Flying object detection mapping (COCO classes that are flying objects)
FLYING_OBJECTS = {
    14: ('bird', 'Bird'),
    15: ('cat', 'Cat'),  # Can fly up
    16: ('dog', 'Dog'),  # Can jump up
    55: ('airplane', 'Aeroplane'),
    60: ('bus', 'Bus'),  # Not flying but large
    61: ('train', 'Train'),  # Not flying
    62: ('truck', 'Truck'),  # Not flying
    # Drones/helicopters might be detected as general objects
    # We'll add custom mapping
}

# Custom flying object classes with their specific BGR colors
FLYING_OBJECT_CLASSES = {
    0: ('bird', 'Bird Detected', (128, 128, 128)), # Gray
    1: ('airplane', 'Airplane Detected', (0, 0, 255)),     # Red
    2: ('helicopter', 'Helicopter Detected', (19, 69, 139)), # Brown
    3: ('drone', 'Drone Detected', (203, 192, 255)), # Pink
}

# Model selection — prefer the custom-trained model, fall back to generic YOLOv8n
_CUSTOM_BEST = Path("captured_datasets/training_results/custom_drone_detector/weights/best.pt")
# Use consistent forward slashes for matching in UI
MODEL_PATH = _CUSTOM_BEST.as_posix() if _CUSTOM_BEST.exists() else "yolov8n.pt"
USING_CUSTOM_MODEL = _CUSTOM_BEST.exists()

logger_tmp = logging.getLogger(__name__)
if USING_CUSTOM_MODEL:
    logger_tmp.info(f"🎯 Custom model found: {MODEL_PATH}")
else:
    logger_tmp.info("ℹ️  Custom model not found, using yolov8n.pt")

# COCO class IDs → our custom class IDs (used by the fallback yolov8n.pt detector)
COCO_TO_CUSTOM_CLASS = {
    4:  2,   # airplane  → class 2
    14: 4,   # bird      → class 4
}
COCO_FLYING_CLASSES = set(COCO_TO_CUSTOM_CLASS.keys())  # only these trigger capture

def list_available_models():
    """List all available YOLOv8 models including yolov8n.pt and trained versions."""
    models = []
    
    # 1. Base model
    if Path("yolov8n.pt").exists():
        models.append({
            'name': 'YOLOv8 Nano (Base)',
            'path': 'yolov8n.pt',
            'type': 'base'
        })
    
    # 2. Search for trained models in captured_datasets/training_results/
    training_results_dir = Path("captured_datasets/training_results")
    if training_results_dir.exists():
        # Look for best.pt in all subdirectories (using forward slashes via as_posix())
        for best_weight in sorted(training_results_dir.glob("**/weights/best.pt")):
            # Get the experiment name (the directory above 'weights')
            experiment_name = best_weight.parent.parent.name
            models.append({
                'name': f"Trained: {experiment_name}",
                'path': best_weight.as_posix(),
                'type': 'custom'
            })
            
    return models


class FrameManager:
    """Manages incoming frames and capture state."""
    
    def __init__(self):
        self.current_frame = None
        self.current_frame_id = 0
        self.current_detections = []  # Store current frame's detections
        self.capture_mode = False  # Continuous capture
        self.connected_clients = []
        
        # Capture statistics
        self.detected_objects_count = 0
        self.background_frames_count = 0
        
        # Paths - New structure with frames/ and labels/ subdirectories
        self.base_dir = Path("captured_datasets")
        
        # Detected objects dataset
        self.detected_dir = self.base_dir / "detected_objects"
        self.detected_frames_dir = self.detected_dir / "frames"
        self.detected_labels_dir = self.detected_dir / "labels"
        
        # Background frames dataset
        self.background_dir = self.base_dir / "background_frames"
        self.background_frames_dir = self.background_dir / "frames"
        self.background_labels_dir = self.background_dir / "labels"
        
        # Combined frames folder (all frames regardless of type)
        self.all_frames_dir = self.base_dir / "frames"
        
        # Create all directories
        self.detected_frames_dir.mkdir(parents=True, exist_ok=True)
        self.detected_labels_dir.mkdir(parents=True, exist_ok=True)
        self.background_frames_dir.mkdir(parents=True, exist_ok=True)
        self.background_labels_dir.mkdir(parents=True, exist_ok=True)
        self.all_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # YOLOv8 detector
        self.detector = None
        self.fallback_detector = None  # yolov8n.pt for reliable detection when custom model misses
        self.sahi_model = None

        if SAHI_AVAILABLE:
            try:
                logger.info(f"Loading SAHI model: {MODEL_PATH}")
                self.sahi_model = UltralyticsDetectionModel(
                    model_path=MODEL_PATH,
                    confidence_threshold=0.3,
                    device="cpu"
                )
                logger.info("✓ SAHI model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load SAHI: {e}")
        elif YOLO_AVAILABLE:
            try:
                logger.info(f"Loading primary model: {MODEL_PATH}")
                self.detector = YOLO(MODEL_PATH)
                logger.info(f"✓ Primary model loaded ({MODEL_PATH})")
            except Exception as e:
                logger.warning(f"Could not load primary model: {e}")

        # Always load yolov8n.pt as fallback when using a custom model
        # (custom model may not detect anything if still in early training)
        if USING_CUSTOM_MODEL and YOLO_AVAILABLE:
            try:
                logger.info("Loading fallback detector: yolov8n.pt")
                self.fallback_detector = YOLO("yolov8n.pt")
                logger.info("✓ Fallback detector loaded (yolov8n.pt)")
            except Exception as e:
                logger.warning(f"Could not load fallback detector: {e}")

        # If primary model failed entirely, use yolov8n.pt as the primary
        if self.detector is None and self.sahi_model is None and YOLO_AVAILABLE:
            try:
                self.detector = YOLO("yolov8n.pt")
                logger.info("✓ Using yolov8n.pt as primary (no custom model)")
            except Exception as e:
                logger.warning(f"Could not load any model: {e}")

    def switch_model(self, new_model_path: str):
        """Reload the detection model with new weights."""
        try:
            logger.info(f"🔄 Switching model to: {new_model_path}")
            
            # Clear existing models
            self.detector = None
            self.sahi_model = None
            
            # Update global state for future references
            global MODEL_PATH, USING_CUSTOM_MODEL
            MODEL_PATH = new_model_path
            USING_CUSTOM_MODEL = ("best.pt" in new_model_path.lower())
            
            if SAHI_AVAILABLE:
                try:
                    self.sahi_model = UltralyticsDetectionModel(
                        model_path=new_model_path,
                        confidence_threshold=0.3,
                        device="cpu"
                    )
                    logger.info("✓ Model switched successfully (SAHI mode)")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load SAHI model: {e}, trying standard YOLO")
            
            if YOLO_AVAILABLE:
                self.detector = YOLO(new_model_path)
                logger.info(f"✓ Model switched successfully (Standard YOLO mode)")
                return True
                
            return False
                
            return False
        except Exception as e:
            logger.error(f"❌ Error switching model: {e}")
            return False



    def process_frame(self, frame: np.ndarray) -> dict:
        """Process frame for detection using custom model, with yolov8n.pt fallback."""
        self.current_frame = frame.copy()
        self.current_frame_id += 1

        detections = []
        has_object = False

        # ---- Primary detection (custom model or SAHI) ----
        if self.sahi_model:
            try:
                results = get_sliced_prediction(
                    image=frame,
                    detection_model=self.sahi_model,
                    slice_height=256,
                    slice_width=256,
                    overlap_height_ratio=0.1,
                    overlap_width_ratio=0.1
                )
                has_object = len(results.object_prediction_list) > 0
                for obj in results.object_prediction_list:
                    bbox = obj.bbox.to_xyxy()
                    detections.append({
                        'class': int(obj.category.id) if obj.category.id else 0,
                        'class_name': obj.category.name if obj.category.name else 'unknown',
                        'confidence': float(obj.score.value) if obj.score else 0.0,
                        'bbox': {'x1': float(bbox[0]), 'y1': float(bbox[1]),
                                 'x2': float(bbox[2]), 'y2': float(bbox[3])},
                        'center': {'x': float((bbox[0]+bbox[2])/2), 'y': float((bbox[1]+bbox[3])/2)},
                        'width': float(bbox[2]-bbox[0]), 'height': float(bbox[3]-bbox[1])
                    })
            except Exception as e:
                logger.error(f"SAHI detection error: {e}")

        elif self.detector:
            try:
                results = self.detector(frame, verbose=False)
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    
                    # If this is a base COCO model, map classes to our internal ones
                    if MODEL_PATH == "yolov8n.pt" and class_id in COCO_TO_CUSTOM_CLASS:
                        class_id = COCO_TO_CUSTOM_CLASS[class_id]
                    
                    class_name = self.detector.names.get(class_id, f'class_{class_id}')
                    bbox = box.xyxy[0].tolist()
                    detections.append({
                        'class': class_id,
                        'class_name': class_name,
                        'confidence': float(box.conf[0]),
                        'bbox': {'x1': float(bbox[0]), 'y1': float(bbox[1]),
                                 'x2': float(bbox[2]), 'y2': float(bbox[3])},
                        'center': {'x': float((bbox[0]+bbox[2])/2), 'y': float((bbox[1]+bbox[3])/2)},
                        'width': float(bbox[2]-bbox[0]), 'height': float(bbox[3]-bbox[1])
                    })
                has_object = len(detections) > 0
            except Exception as e:
                logger.error(f"Detection error: {e}")

        # ---- Fallback: yolov8n.pt with COCO flying-object filter ----
        # Used when the custom model is still too new and detects nothing.
        if not has_object and self.fallback_detector:
            try:
                fb_results = self.fallback_detector(frame, verbose=False)
                for box in fb_results[0].boxes:
                    coco_class_id = int(box.cls[0])
                    if coco_class_id not in COCO_FLYING_CLASSES:
                        continue  # ignore non-flying objects
                    custom_class_id = COCO_TO_CUSTOM_CLASS[coco_class_id]
                    class_name = FLYING_OBJECT_CLASSES.get(custom_class_id, (f'class_{custom_class_id}',))[0]
                    bbox = box.xyxy[0].tolist()
                    detections.append({
                        'class': custom_class_id,
                        'class_name': class_name,
                        'confidence': float(box.conf[0]),
                        'bbox': {'x1': float(bbox[0]), 'y1': float(bbox[1]),
                                 'x2': float(bbox[2]), 'y2': float(bbox[3])},
                        'center': {'x': float((bbox[0]+bbox[2])/2), 'y': float((bbox[1]+bbox[3])/2)},
                        'width': float(bbox[2]-bbox[0]), 'height': float(bbox[3]-bbox[1])
                    })
                if detections:
                    has_object = True
                    logger.debug(f"Fallback detector found {len(detections)} flying object(s)")
            except Exception as e:
                logger.error(f"Fallback detection error: {e}")

        # Store detections for later use
        self.current_detections = detections

        # ===== DEBUG =====
        logger.info(f"Found {len(detections)} detections in frame {self.current_frame_id}")

        for d in detections:
            logger.info(d)
        # =================
        return {
            'frame_id': self.current_frame_id,
            'has_object': has_object,
            'detections': detections,
            'timestamp': datetime.utcnow().isoformat()
        }

    def capture_detected_frame(self, detections: list = None) -> bool:
        """Save detected object frame with location metadata in YOLO format."""
        if self.current_frame is None:
            return False
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename_base = f"object_{timestamp}_{self.current_frame_id:06d}"
            
            # Save image to detected_objects/frames/
            image_path = self.detected_frames_dir / f"{filename_base}.jpg"
            success = cv2.imwrite(str(image_path), self.current_frame)
            
            if not success:
                logger.error(f"Failed to save image: {filename_base}")
                return False
            
            logger.info(f"✓ Saved frame to: {image_path}")
            
            # Also save to combined frames folder
            combined_image_path = self.all_frames_dir / f"{filename_base}.jpg"
            cv2.imwrite(str(combined_image_path), self.current_frame)
            
            # Save detection labels in YOLO format to detected_objects/labels/
            if detections:
                txt_path = self.detected_labels_dir / f"{filename_base}.txt"
                img_height, img_width = self.current_frame.shape[:2]
                
                with open(txt_path, 'w') as f:
                    for det in detections:
                        # Extract detection info - support both 'class' and 'class_id' keys
                        class_id = int(det.get('class_id', det.get('class', 0)))
                        bbox = det.get('bbox', {})
                        
                        # Get coordinates
                        x1 = float(bbox.get('x1', 0))
                        y1 = float(bbox.get('y1', 0))
                        x2 = float(bbox.get('x2', img_width))
                        y2 = float(bbox.get('y2', img_height))
                        
                        # Convert to YOLO format (normalized center x, y, width, height)
                        center_x = (x1 + x2) / 2.0 / img_width
                        center_y = (y1 + y2) / 2.0 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        # Clamp values to 0-1 range
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        # Write in YOLO format: class_id center_x center_y width height
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                logger.info(f"✓ Captured object frame: {filename_base} ({len(detections)} objects)")
            else:
                logger.info(f"✓ Captured object frame: {filename_base}")
            
            self.detected_objects_count += 1
            return True
        
        except Exception as e:
            logger.error(f"Error saving detected frame: {e}")
            return False

    def capture_background_frame(self, detections: list = None) -> bool:
        """Save background frame with detection metadata in YOLO format."""
        if self.current_frame is None:
            return False
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename_base = f"background_{timestamp}_{self.current_frame_id:06d}"
            
            # Save image to background_frames/frames/
            image_path = self.background_frames_dir / f"{filename_base}.jpg"
            success = cv2.imwrite(str(image_path), self.current_frame)
            
            if not success:
                logger.error(f"Failed to save image: {filename_base}")
                return False
            
            logger.info(f"✓ Saved background frame to: {image_path}")
            
            # Also save to combined frames folder
            combined_image_path = self.all_frames_dir / f"{filename_base}.jpg"
            cv2.imwrite(str(combined_image_path), self.current_frame)
            
            # Save detection labels in YOLO format to background_frames/labels/
            txt_path = self.background_labels_dir / f"{filename_base}.txt"
            img_height, img_width = self.current_frame.shape[:2]
            
            with open(txt_path, 'w') as f:
                if detections:
                    for det in detections:
                        # Extract detection info - support both 'class' and 'class_id' keys
                        class_id = int(det.get('class_id', det.get('class', 0)))
                        bbox = det.get('bbox', {})
                        
                        # Get coordinates
                        x1 = float(bbox.get('x1', 0))
                        y1 = float(bbox.get('y1', 0))
                        x2 = float(bbox.get('x2', img_width))
                        y2 = float(bbox.get('y2', img_height))
                        
                        # Convert to YOLO format (normalized center x, y, width, height)
                        center_x = (x1 + x2) / 2.0 / img_width
                        center_y = (y1 + y2) / 2.0 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        # Clamp values to 0-1 range
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        # Write in YOLO format: class_id center_x center_y width height
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            logger.info(f"✓ Captured background frame: {filename_base}")
            self.background_frames_count += 1
            logger.info(f"Background frames saved to: {self.background_frames_dir}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving background frame: {e}")
            import traceback
            traceback.print_exc()
            return False

    def capture_detected_frame_with_labels(self, detections: list = None, original_frame = None) -> bool:
        """Save detected object frame with bounding boxes drawn and TXT file with object metadata."""
        try:
            # Validate frame
            if original_frame is None:
                original_frame = self.current_frame
            
            if original_frame is None:
                logger.error("❌ No frame to capture - both original_frame and current_frame are None")
                return False
            
            logger.info(f"📸 Attempting to capture frame...")
            logger.info(f"  Frame shape: {original_frame.shape}")
            logger.info(f"  Detections received: {len(detections) if detections else 0}")
            
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            frame_id = self.current_frame_id
            filename_base = f"object_{timestamp}_{frame_id:06d}"
            logger.info(f"  Filename: {filename_base}")
            
            # Step 1: Draw bounding boxes on frame
            try:
                frame_with_boxes = draw_flying_objects_on_frame(original_frame.copy(), detections if detections else [])
                logger.info(f"✓ Drawn bounding boxes on frame")
            except Exception as draw_error:
                logger.error(f"⚠️ Error drawing boxes: {draw_error}, saving frame without boxes")
                frame_with_boxes = original_frame.copy()
            
            # Step 2: Save JPG image
            try:
                image_path = self.detected_frames_dir / f"{filename_base}.jpg"
                logger.info(f"  Saving to: {image_path}")
                
                success = cv2.imwrite(str(image_path), frame_with_boxes)
                if not success:
                    logger.error(f"❌ cv2.imwrite returned False")
                    return False
                
                file_size = image_path.stat().st_size
                logger.info(f"✓ Frame saved ({file_size} bytes) to: {image_path.name}")
            except Exception as img_error:
                logger.error(f"❌ Failed to save image: {img_error}")
                import traceback
                traceback.print_exc()
                return False
            
            # Step 3: Save TXT metadata file
            try:
                txt_path = self.detected_labels_dir / f"{filename_base}.txt"
                logger.info(f"  Metadata file: {txt_path}")
                
                with open(txt_path, 'w') as f:
                    if detections and len(detections) > 0:
                        for idx, det in enumerate(detections):
                            try:
                                # Safely extract values with defaults
                                class_id = int(det.get('class_id', det.get('class', 0)))
                                bbox = det.get('bbox', {})
                                confidence = float(det.get('confidence', 0.0))
                                
                                # Get coordinates
                                x1 = float(bbox.get('x1', 0))
                                y1 = float(bbox.get('y1', 0))
                                x2 = float(bbox.get('x2', original_frame.shape[1]))
                                y2 = float(bbox.get('y2', original_frame.shape[0]))
                                
                                # Calculate center and dimensions
                                center_x = (x1 + x2) / 2.0
                                center_y = (y1 + y2) / 2.0
                                width = abs(x2 - x1)
                                height = abs(y2 - y1)
                                
                                # Get object label
                                object_name = det.get('class_name', FLYING_OBJECT_CLASSES.get(class_id % len(FLYING_OBJECT_CLASSES), ('unknown', 'Unknown'))[0])
                                
                                # Write metadata
                                line = f"{class_id} {center_x:.2f} {center_y:.2f} {width:.2f} {height:.2f} {object_name} {confidence:.4f}\n"
                                f.write(line)
                                logger.info(f"  ✓ Object {idx+1}: class={class_id}, center=({center_x:.1f},{center_y:.1f}), size={width:.1f}x{height:.1f}")
                            except Exception as det_error:
                                logger.error(f"  ⚠️ Error processing detection {idx}: {det_error}")
                        logger.info(f"✓ Metadata file saved with {len(detections)} objects")
                    else:
                        # Write completely empty file
                        pass
                        logger.info(f"✓ Metadata file saved (empty - no detections)")
                
            except Exception as txt_error:
                logger.error(f"❌ Failed to save metadata: {txt_error}")
                import traceback
                traceback.print_exc()
                # Don't return false - image was saved successfully
            
            # Success!
            self.detected_objects_count += 1
            logger.info(f"✅ FRAME CAPTURED successfully! (Total: {self.detected_objects_count})")
            logger.info(f"   📁 Saved to: captured_datasets/detected_objects/")
            return True
        
        except Exception as e:
            logger.error(f"❌ Unexpected error in capture_detected_frame_with_labels: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_stats(self) -> dict:
        """Get capture statistics."""
        return {
            'detected_objects': self.detected_objects_count,
            'background_frames': self.background_frames_count,
            'total': self.detected_objects_count + self.background_frames_count,
            'current_frame_id': self.current_frame_id,
            'capture_mode': self.capture_mode
        }


frame_manager = FrameManager()

# ==============================================================
# HELPER FUNCTIONS FOR FLYING OBJECT DETECTION & DRAWING
# ==============================================================

def is_flying_object(detection: dict) -> bool:
    """Check if detection is a flying object."""
    # For now, accept all detections as they could be flying objects
    # You can add specific filtering logic here
    class_id = detection.get('class', -1)
    # If you have a trained model, you can check against specific classes
    return True  # Accept all for now

def draw_flying_objects_on_frame(frame, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on frame for flying objects with specific colors."""
    frame_copy = frame.copy()
    img_height, img_width = frame.shape[:2]
    
    for det in detections:
        if not is_flying_object(det):
            continue
        
        # Get class info
        class_id = det.get('class', 0)
        confidence = det.get('confidence', 0.0)
        
        # Get bounding box
        bbox = det.get('bbox', {})
        x1 = int(bbox.get('x1', 0))
        y1 = int(bbox.get('y1', 0))
        x2 = int(bbox.get('x2', img_width))
        y2 = int(bbox.get('y2', img_height))
        
        # Determine color and label based on class
        # Ensure class_id is within our mapped range
        lookup_id = class_id % len(FLYING_OBJECT_CLASSES)
        class_info = FLYING_OBJECT_CLASSES.get(lookup_id, ('unknown', 'Unknown', (255, 255, 255)))
        
        object_label = class_info[1]
        box_color = class_info[2]
        
        # 1. Draw "ENCIRCLE" - a prominent ring around the centroid
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = int(max(x2 - x1, y2 - y1) * 0.7)  # Slightly larger than object
        cv2.circle(frame_copy, (center_x, center_y), radius, box_color, 3) # Outer ring
        cv2.circle(frame_copy, (center_x, center_y), 5, box_color, -1)     # Center dot
        
        # 2. Draw standard bounding box (requested as "respective box colors")
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), box_color, 2)
        
        # 3. Create label with name and confidence
        label_text = f"{object_label} {confidence:.1%}"
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        # Get text size
        (label_width, label_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        # Draw background box for text
        cv2.rectangle(frame_copy, (x1, y1 - label_height - 10), (x1 + label_width + 10, y1), box_color, -1)
        
        # Draw text in white (or black for yellow birds)
        text_color = (0, 0, 0) if lookup_id == 4 else (255, 255, 255)
        cv2.putText(frame_copy, label_text, (x1 + 5, y1 - 5),
                   font, font_scale, text_color, font_thickness)
    
    return frame_copy

# ==============================================================
# VIDEO PROCESSING FUNCTIONS
# ==============================================================

def list_videos_in_dataset():
    """List all video files in the input_dataset folder."""
    try:
        videos = []
        if VIDEO_DATASET_DIR.exists():
            for video_file in VIDEO_DATASET_DIR.iterdir():
                if video_file.is_file() and video_file.suffix.lower() in VIDEO_EXTENSIONS:
                    videos.append({
                        'name': video_file.name,
                        'path': str(video_file),
                        'size': video_file.stat().st_size,
                        'size_mb': round(video_file.stat().st_size / (1024 * 1024), 2)
                    })
        return sorted(videos, key=lambda x: x['name'])
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        return []

def process_video_frames(video_path: str):
    """Extract frames from video and display them. Only capture on user button click."""
    global video_processing_active
    
    try:
        logger.info(f"Starting video processing: {video_path}")
        
        # Do NOT auto-enable capture mode - wait for user to click button
        frame_manager.capture_mode = False
        logger.info(f"↓ Capture mode initialized as FALSE - waiting for user to click CAPTURE FRAME button")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            video_processing_active = False
            return
        
        frame_count = 0
        capture_attempts = 0
        successful_captures = 0
        
        while video_processing_active:
            ret, frame = cap.read()
            if not ret:
                logger.info(f"✓ Video processing completed:")
                logger.info(f"  - Total frames: {frame_count}")
                logger.info(f"  - Capture attempts: {capture_attempts}")
                logger.info(f"  - Successful captures: {successful_captures}")
                logger.info(f"  - Frames in dataset: {frame_manager.detected_objects_count}")
                break
            
            frame_count += 1
            
            # Process frame for detection
            metadata = frame_manager.process_frame(frame)
            
            # Draw detections on frame for display
            frame_with_detections = draw_flying_objects_on_frame(frame, metadata['detections'])
            
            # Replace the current frame with the one that has detections drawn
            frame_manager.current_frame = frame_with_detections
            
            # ONLY capture if user clicked capture AND objects were detected in this frame
            if frame_manager.capture_mode and metadata['has_object']:
                capture_attempts += 1
                logger.debug(f"→ Capture attempt #{capture_attempts} (frame {frame_count}, {len(metadata['detections'])} detections)")
                success = frame_manager.capture_detected_frame_with_labels(detections=metadata['detections'], original_frame=frame)
                if success:
                    successful_captures += 1
                    logger.info(f"✓ Frame {frame_count} captured ({len(metadata['detections'])} objects, total: {successful_captures})")
                else:
                    logger.error(f"✗ Frame {frame_count} capture failed")
            elif frame_manager.capture_mode and not metadata['has_object']:
                logger.debug(f"  Frame {frame_count}: no objects detected, skipping capture")
            
            # Log status every 30 frames
            if frame_count % 30 == 0:
                capture_status = f"Capture ON - {capture_attempts} attempts, {successful_captures} successful"
                logger.info(f"Status: {frame_count} frames processed | {capture_status}")
        
        cap.release()
        logger.info(f"✓ Video processing finished for: {video_path}")
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        video_processing_active = False
        # Do NOT reset capture_mode here — user must explicitly click STOP CAPTURE
        logger.info(f"↓ Video processing thread ended (capture_mode remains: {frame_manager.capture_mode})")

# ==============================================================
# FASTAPI APP
# ==============================================================

app = FastAPI(
    title="Advanced Live Video Streaming Server",
    description="Receives video frames from external source with smart object detection",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================
# ENDPOINTS
# ==============================================================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """HTML dashboard with capture controls."""
    return r"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🛡️ Aerial Object Detection and Classification System</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                min-height: 100vh;
                padding: 20px;
            }
            .main-title {
                text-align: center;
                margin-bottom: 20px;
                text-shadow: 0 2px 10px rgba(0,0,0,0.3);
                font-size: 2em;
                letter-spacing: 1px;
            }
            .container {
                display: flex;
                gap: 20px;
                max-width: 1400px;
                margin: 0 auto;
                align-items: flex-start;
            }
            .left-panel {
                flex: 0 0 320px;
                background: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                padding: 20px;
                border-radius: 12px;
                backdrop-filter: blur(10px);
                height: fit-content;
                position: sticky;
                top: 20px;
            }
            .left-panel h2 {
                margin-bottom: 20px;
                border-bottom: 2px solid rgba(255,255,255,0.3);
                padding-bottom: 10px;
                font-size: 1.2em;
            }
            .right-panel {
                flex: 1;
            }
            .section {
                background: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                padding: 25px;
                border-radius: 12px;
                margin: 20px 0;
                backdrop-filter: blur(10px);
                transition: all 0.3s;
            }
            .section:hover {
                background: rgba(255,255,255,0.15);
            }
            .section h2 {
                margin-bottom: 15px;
                border-bottom: 2px solid rgba(255,255,255,0.3);
                padding-bottom: 10px;
            }
            .video-preview {
                width: 100%;
                max-width: 900px;
                border: 2px solid rgba(255,255,255,0.3);
                border-radius: 8px;
                margin: 15px 0;
            }
            .controls {
                display: flex;
                flex-direction: column;
                gap: 12px;
                margin: 20px 0;
            }
            .action-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-bottom: 15px;
            }
            button {
                padding: 15px 20px;
                border: none;
                border-radius: 8px;
                font-size: 0.95em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                width: 100%;
            }
            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            #captureBtn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: 2px solid transparent;
            }
            #captureBtn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            #captureBtn.active {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                border: 2px solid white;
            }
            #stopBtn {
                background: linear-gradient(135deg, #ea4335 0%, #c5221f 100%);
                color: white;
            }
            #stopBtn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(234, 67, 53, 0.4);
            }
            #backgroundBtn {
                background: linear-gradient(135deg, #34a853 0%, #2d904a 100%);
                color: white;
            }
            #backgroundBtn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(52, 168, 83, 0.4);
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
                margin-top: 15px;
            }
            .stat-card {
                background: rgba(0,0,0,0.3);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.2);
                transition: all 0.3s;
            }
            .stat-card:hover {
                background: rgba(0,0,0,0.5);
                border-color: rgba(255,255,255,0.4);
            }
            .stat-label {
                font-size: 0.75em;
                opacity: 0.8;
                margin-bottom: 8px;
            }
            .stat-value {
                font-size: 2em;
                font-weight: 700;
                color: #4dd0e1;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
                animation: pulse 2s infinite;
            }
            .status-indicator.active {
                background: #34a853;
            }
            .status-indicator.inactive {
                background: #fbbc04;
            }
            .status-indicator.alert {
                background: #ea4335;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            .info-box {
                background: rgba(76, 175, 80, 0.1);
                border-left: 4px solid #4caf50;
                padding: 15px;
                border-radius: 4px;
                margin: 15px 0;
                font-size: 0.9em;
            }
            .info-box ul {
                margin: 10px 0;
                margin-left: 20px;
            }
            .info-box li {
                margin: 5px 0;
            }
            /* Detection History */
            .history-list {
                max-height: 200px;
                overflow-y: auto;
                background: rgba(0,0,0,0.3);
                border-radius: 6px;
                padding: 10px;
                margin-top: 15px;
                font-size: 0.85em;
            }
            .history-item {
                padding: 8px;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .history-item:last-child {
                border-bottom: none;
            }
            .history-item:hover {
                background: rgba(255,255,255,0.1);
            }
            /* Video Selection Styles */
            select {
                width: 100%;
                padding: 12px;
                border: 2px solid rgba(255,255,255,0.3);
                border-radius: 8px;
                font-size: 1em;
                background: rgba(255,255,255,0.1);
                color: white;
                cursor: pointer;
                margin-bottom: 10px;
                font-family: 'Segoe UI', sans-serif;
            }
            select option {
                background: #1e3c72;
                color: white;
            }
            select:hover {
                border-color: rgba(255,255,255,0.5);
                background: rgba(255,255,255,0.15);
            }
            #selectedVideoInfo {
                background: rgba(76, 175, 80, 0.1);
                border-left: 4px solid #4caf50;
                padding: 15px;
                border-radius: 4px;
                margin: 15px 0;
                font-size: 0.9em;
                display: none;
            }
            #videoProgress {
                background: rgba(25, 103, 210, 0.1);
                border-left: 4px solid #1967d2;
                padding: 15px;
                border-radius: 4px;
                margin: 15px 0;
                font-size: 0.9em;
                display: none;
            }
            .model-section {
                background: rgba(102, 126, 234, 0.1);
                border: 1px solid rgba(102, 126, 234, 0.3);
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .model-badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.75em;
                font-weight: 700;
                margin-left: 10px;
                text-transform: uppercase;
            }
            .model-badge.base { background: #4caf50; color: white; }
            .model-badge.custom { background: #ff9800; color: white; }
            
            @media (max-width: 1024px) {
                .container {
                    flex-direction: column-reverse;
                }
                .left-panel {
                    flex: 1;
                    position: static;
                }
            }
        </style>
    </head>
    <body>
        <h1 class="main-title">🛡️ SKYGUARD SMART CAPTURE SYSTEM 📹</h1>
        
        <div class="container">
            <!-- Left Panel - Controls -->
            <div class="left-panel">
                <h2>🎬 CAPTURE CONTROLS</h2>
                <div class="controls">
                    <button id="captureBtn" onclick="toggleCapture()">
                        ▶️ START CAPTURE
                    </button>
                    <button id="stopBtn" onclick="stopCapture()" disabled>
                        ⏹️ STOP CAPTURE
                    </button>
                    <button id="backgroundBtn" onclick="captureBackground()">
                        🖼️ BACKGROUND FRAME
                    </button>
                </div>
                
                <h2 style="margin-top: 30px;">📊 REAL-TIME STATS</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">🎯 Detected</div>
                        <div class="stat-value" id="objectCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🎨 Background</div>
                        <div class="stat-value" id="backgroundCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">📹 Total</div>
                        <div class="stat-value" id="totalCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🔢 Frame ID</div>
                        <div class="stat-value" id="frameId">0</div>
                    </div>
                </div>
                
                <h2 style="margin-top: 30px;">🤖 MODEL SELECTION</h2>
                <div class="model-section">
                    <p style="color: #ccc; margin-bottom: 10px; font-size: 0.85em;">Select detection model version:</p>
                    <select id="modelSelect" onchange="switchModel()">
                        <option value="yolov8n.pt">YOLOv8 Nano (Base)</option>
                    </select>
                    <div id="activeModelInfo" style="font-size: 0.8em; margin-top: 8px; color: #4dd0e1;">
                        Initializing models...
                    </div>
                </div>

                <h2 style="margin-top: 30px;">📜 DETECTION HISTORY</h2>
                <div class="history-list" id="historyList">
                    <div class="history-item" style="display: flex; justify-content: space-between;">
                        <span>✨ System initialized</span>
                        <span style="font-size: 0.75em;">Now</span>
                    </div>
                </div>
            </div>
            
            <!-- Right Panel - Video and Info -->
            <div class="right-panel">
                <div class="section">
                    <h2>📺 LIVE VIDEO FEED</h2>
                    <img class="video-preview" id="videoPreview" src="/live-stream" alt="Live Video Feed">
                    <div style="text-align: center; margin-top: 10px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 6px;">
                        <span class="status-indicator" id="statusIndicator"></span>
                        <span id="connectionStatus">🔴 Waiting for frames...</span>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📽️ DATASET VIDEO PROCESSOR</h2>
                    <p style="color: #ccc; margin-bottom: 15px; font-size: 0.9em;">🎥 Select & analyze videos from your input_dataset folder</p>
                    <label style="display: block; margin-bottom: 10px; color: #fff; font-weight: 600;">🗂️ Available Videos:</label>
                    <select id="videoSelect">
                        <option value="">-- Select a video --</option>
                    </select>
                    
                    <div id="selectedVideoInfo">
                        <div style="color: #4caf50; font-weight: 600; margin-bottom: 8px;">✅ VIDEO SELECTED</div>
                        <div style="color: #fff; font-size: 0.95em;">
                            <div><strong>📄 Name:</strong> <span id="selectedVideoName"></span></div>
                            <div><strong>💾 Size:</strong> <span id="selectedVideoSize"></span></div>
                        </div>
                    </div>
                    
                    <div class="action-grid">
                        <button id="processVideoBtn" onclick="startVideoProcess()" disabled style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">▶️ ANALYZE</button>
                        <button id="stopVideoBtn" onclick="stopVideoProcess()" disabled style="background: linear-gradient(135deg, #ea4335 0%, #c5221f 100%); color: white;">⏹️ STOP</button>
                    </div>
                    
                    <div id="videoProgress">
                        <div style="color: #1967d2; font-weight: 600; margin-bottom: 10px;">⏳ PROCESSING STATUS:</div>
                        <div id="videoProgressText" style="color: #fff;">Ready...</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>💡 SYSTEM INFORMATION</h2>
                    <div class="info-box">
                        <ul>
                            <li><strong>🎬 START CAPTURE:</strong> Begin capturing detected flying objects</li>
                            <li><strong>🤖 AUTO-DETECT:</strong> YOLOv8 + SAHI dual-mode detection</li>
                            <li><strong>📍 OBJECT STORAGE:</strong> <code>captured_datasets/detected_objects/</code></li>
                            <li><strong>🎨 BACKGROUND CAPTURE:</strong> Train-set background frames</li>
                            <li><strong>📦 BACKGROUND STORAGE:</strong> <code>captured_datasets/background_frames/</code></li>
                            <li><strong>📝 METADATA:</strong> JSON + bounding boxes for each capture</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // === MODEL MANAGEMENT ===
            async function loadAvailableModels() {
                try {
                    const response = await fetch('/list-models');
                    const data = await response.json();
                    
                    if (data.success) {
                        const modelSelect = document.getElementById('modelSelect');
                        modelSelect.innerHTML = ''; // Clear
                        
                        data.models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model.path;
                            option.textContent = model.name;
                            if (model.path === data.current_model) {
                                option.selected = true;
                            }
                            modelSelect.appendChild(option);
                        });
                        
                        const currentName = data.models.find(m => m.path === data.current_model)?.name || 'Unknown';
                        document.getElementById('activeModelInfo').textContent = `Active: ${currentName}`;
                        console.log('✓ Models loaded');
                    }
                } catch (e) {
                    console.error('Error loading models:', e);
                }
            }

            async function switchModel() {
                const modelSelect = document.getElementById('modelSelect');
                const modelPath = modelSelect.value;
                const modelName = modelSelect.options[modelSelect.selectedIndex].text;
                
                try {
                    document.getElementById('activeModelInfo').textContent = `Switching to ${modelName}...`;
                    addToHistory(`Switching model to ${modelName}`, '🔄');
                    
                    const response = await fetch('/switch-model', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ model_path: modelPath })
                    });
                    
                    const data = await response.json();
                    if (data.success) {
                        document.getElementById('activeModelInfo').textContent = `Active: ${modelName}`;
                        addToHistory(`Active model: ${modelName}`, '✅');
                    } else {
                        alert(`Failed to switch model: ${data.message}`);
                        addToHistory(`Model switch failed`, '⚠️');
                        loadAvailableModels(); // Reset selection
                    }
                } catch (e) {
                    console.error('Error switching model:', e);
                    alert('Error switching model');
                    loadAvailableModels(); // Reset selection
                }
            }

            // === HISTORY TRACKING ===
            function addToHistory(message, emoji = '📌') {
                const historyList = document.getElementById('historyList');
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                const timestamp = new Date().toLocaleTimeString();
                historyItem.innerHTML = `
                    <span>${emoji} ${message}</span>
                    <span style="font-size: 0.75em;">${timestamp}</span>
                `;
                historyList.insertBefore(historyItem, historyList.firstChild);
                
                // Keep only last 15 items
                while (historyList.children.length > 15) {
                    historyList.removeChild(historyList.lastChild);
                }
            }
            
            // === VIDEO DATASET HANDLING ===
            const videoSelect = document.getElementById('videoSelect');
            const selectedVideoInfo = document.getElementById('selectedVideoInfo');
            const processVideoBtn = document.getElementById('processVideoBtn');
            const stopVideoBtn = document.getElementById('stopVideoBtn');
            const videoProgress = document.getElementById('videoProgress');
            let videoProcessingActive = false;
            
            // Load available videos when page loads
            async function loadAvailableVideos() {
                try {
                    const response = await fetch('/list-videos');
                    const data = await response.json();
                    
                    if (data.success && data.videos && data.videos.length > 0) {
                        // Clear existing options except the first one
                        while (videoSelect.options.length > 1) {
                            videoSelect.remove(1);
                        }
                        
                        // Add video options
                        data.videos.forEach(video => {
                            const option = document.createElement('option');
                            option.value = video.name;
                            option.textContent = `🎬 ${video.name} (${video.size_mb} MB)`;
                            videoSelect.appendChild(option);
                        });
                        
                        console.log(`✓ Loaded ${data.total} video(s) from dataset`);
                        addToHistory(`Loaded ${data.total} video(s)`, '📥');
                    }
                } catch (e) {
                    console.error('Error loading videos:', e);
                    addToHistory('Failed to load videos', '⚠️');
                }
            }
            
            // Handle video selection
            videoSelect.addEventListener('change', (e) => {
                const selectedValue = e.target.value;
                
                if (selectedValue) {
                    const selectedOption = videoSelect.options[videoSelect.selectedIndex];
                    const optionText = selectedOption.textContent;
                    const videoName = selectedValue;
                    
                    // Extract size from option text
                    const sizeMatch = optionText.match(/\(([^)]+)\)/);
                    const sizeText = sizeMatch ? sizeMatch[1] : 'Unknown';
                    
                    // Show selected video info
                    document.getElementById('selectedVideoName').textContent = videoName;
                    document.getElementById('selectedVideoSize').textContent = sizeText;
                    selectedVideoInfo.style.display = 'block';
                    processVideoBtn.disabled = false;
                    
                    console.log(`Video selected: ${videoName}`);
                    addToHistory(`Selected: ${videoName}`, '🎥');
                } else {
                    selectedVideoInfo.style.display = 'none';
                    processVideoBtn.disabled = true;
                    videoProgress.style.display = 'none';
                }
            });
            
            // Process video button
            async function startVideoProcess() {
                const selectedVideo = videoSelect.value;
                
                if (!selectedVideo) {
                    alert('⚠️ Please select a video first');
                    return;
                }
                
                try {
                    processVideoBtn.disabled = true;
                    videoProgress.style.display = 'block';
                    videoProcessingActive = true;
                    
                    console.log(`⏳ Starting video analysis: ${selectedVideo}...`);
                    addToHistory(`Starting analysis...`, '⏳');
                    
                    const response = await fetch('/process-video', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            video_name: selectedVideo
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        console.log(`✓ Video analysis started: ${selectedVideo}`);
                        addToHistory(`Analysis started: ${selectedVideo}`, '▶️');
                        stopVideoBtn.disabled = false;
                        updateVideoProgress();
                    } else {
                        alert(`✗ Error: ${data.message}`);
                        addToHistory(`Error: ${data.message}`, '❌');
                        processVideoBtn.disabled = false;
                        videoProgress.style.display = 'none';
                        videoProcessingActive = false;
                    }
                } catch (error) {
                    alert(`✗ Error starting video processing: ${error.message}`);
                    addToHistory(`Error: ${error.message}`, '❌');
                    console.error('Video processing error:', error);
                    processVideoBtn.disabled = false;
                    videoProgress.style.display = 'none';
                    videoProcessingActive = false;
                }
            }
            
            // Stop video button
            async function stopVideoProcess() {
                try {
                    const response = await fetch('/stop-video', {
                        method: 'POST'
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        console.log('✓ Video analysis stopped');
                        addToHistory('Analysis stopped', '⏹️');
                        videoProcessingActive = false;
                        stopVideoBtn.disabled = true;
                        processVideoBtn.disabled = false;
                        videoProgress.style.display = 'none';
                    }
                } catch (error) {
                    alert(`✗ Error stopping video: ${error.message}`);
                    console.error('Stop video error:', error);
                    addToHistory(`Stop error: ${error.message}`, '❌');
                }
            }
            
            // Update video processing progress
            function updateVideoProgress() {
                const updateInterval = setInterval(async () => {
                    if (!videoProcessingActive) {
                        clearInterval(updateInterval);
                        return;
                    }
                    
                    try {
                        const response = await fetch('/stats');
                        const stats = await response.json();
                        
                        const progressText = document.getElementById('videoProgressText');
                        progressText.innerHTML = `
                            <div>📊 Frames Processed: <strong>${stats.current_frame_id}</strong></div>
                            <div>🎯 Objects Detected: <strong>${stats.detected_objects}</strong></div>
                            <div>🎨 Background Frames: <strong>${stats.background_frames}</strong></div>
                        `;
                    } catch (e) {
                        console.error('Error updating progress:', e);
                    }
                }, 2000);
            }
            
            // Load data on page load
            loadAvailableVideos();
            loadAvailableModels();
            
            // Refresh lists periodically
            setInterval(loadAvailableVideos, 10000);
            setInterval(loadAvailableModels, 15000);
            
            let captureActive = false;
            let lastFrameId = -1;
            let lastFrameTime = Date.now();
            let staleWarningShown = false;
            
            function toggleCapture() {
                captureActive = !captureActive;
                const btn = document.getElementById('captureBtn');
                const stopBtn = document.getElementById('stopBtn');
                
                if (captureActive) {
                    fetch('/capture-start', { method: 'POST' });
                    btn.classList.add('active');
                    btn.textContent = '⏸️ CAPTURING...';
                    stopBtn.disabled = false;
                    addToHistory('Capture started', '🔴');
                } else {
                    fetch('/capture-stop', { method: 'POST' });
                    btn.classList.remove('active');
                    btn.textContent = '▶️ START CAPTURE';
                    stopBtn.disabled = true;
                    addToHistory('Capture paused', '⏸️');
                }
            }
            
            function stopCapture() {
                captureActive = false;
                fetch('/capture-stop', { method: 'POST' });
                const btn = document.getElementById('captureBtn');
                const stopBtn = document.getElementById('stopBtn');
                btn.classList.remove('active');
                btn.textContent = '▶️ START CAPTURE';
                stopBtn.disabled = true;
                addToHistory('Capture stopped', '⏹️');
            }
            
            function captureBackground() {
                fetch('/capture-background', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        if (data.success) {
                            addToHistory('Background frame captured', '🖼️');
                        } else {
                            addToHistory('Background capture failed', '⚠️');
                        }
                    });
            }
            
            // ==== Sync button state with server (handles server restarts) ====
            function syncCaptureButton(serverCaptureMode) {
                if (serverCaptureMode === captureActive) return; // already in sync
                captureActive = serverCaptureMode;
                const btn = document.getElementById('captureBtn');
                const stopBtn = document.getElementById('stopBtn');
                if (captureActive) {
                    btn.classList.add('active');
                    btn.textContent = '⏸️ CAPTURING...';
                    stopBtn.disabled = false;
                } else {
                    btn.classList.remove('active');
                    btn.textContent = '▶️ START CAPTURE';
                    stopBtn.disabled = true;
                }
            }

            // Update stats every 2 seconds
            setInterval(async () => {
                try {
                    const response = await fetch('/stats');
                    const data = await response.json();

                    document.getElementById('objectCount').textContent = data.detected_objects;
                    document.getElementById('backgroundCount').textContent = data.background_frames;
                    document.getElementById('totalCount').textContent = data.total;
                    document.getElementById('frameId').textContent = data.current_frame_id;

                    // Sync button with server state (critical after server restart)
                    syncCaptureButton(data.capture_mode);

                    // Detect stale frames (no new frames for > 3 seconds)
                    if (data.current_frame_id !== lastFrameId) {
                        lastFrameId = data.current_frame_id;
                        lastFrameTime = Date.now();
                        staleWarningShown = false;
                    }
                    const stale = (Date.now() - lastFrameTime) > 3000 && data.current_frame_id > 0;

                    // Update status indicator
                    const indicator = document.getElementById('statusIndicator');
                    const status = document.getElementById('connectionStatus');

                    if (stale && !staleWarningShown) {
                        staleWarningShown = true;
                        indicator.className = 'status-indicator inactive';
                        status.textContent = '⚠️ No new frames — start frame_sender or video processor';
                        if (data.capture_mode) addToHistory('⚠️ Capture ON but no frames flowing in', '⚠️');
                    } else if (!stale) {
                        if (data.capture_mode) {
                            indicator.className = 'status-indicator alert';
                            status.textContent = '🔴 RECORDING - Capturing frames';
                        } else if (data.current_frame_id > 0) {
                            indicator.className = 'status-indicator active';
                            status.textContent = '🟢 Connected - Ready to capture';
                        }
                    }
                } catch (e) {
                    console.log('Stats fetch error:', e);
                }
            }, 2000);

            // Sync on page load immediately
            (async () => {
                try {
                    const r = await fetch('/stats');
                    const d = await r.json();
                    syncCaptureButton(d.capture_mode);
                } catch(e) {}
            })();
        </script>
    </body>
    </html>
    """


@app.post("/submit-frame")
async def submit_frame(frame: UploadFile = File(...)):
    """Accept frame from external program."""
    try:
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame_data is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Process frame for detection
        metadata = frame_manager.process_frame(frame_data)
        
        # Capture if mode is active and object detected
        if frame_manager.capture_mode and metadata['has_object']:
            frame_manager.capture_detected_frame(detections=metadata['detections'])
        
        return {
            'success': True,
            'frame_id': metadata['frame_id'],
            'has_object': metadata['has_object'],
            'detections_count': len(metadata['detections']),
            'detections': metadata['detections']
        }
    
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/capture-start")
async def capture_start():
    """Start continuous capture mode."""
    frame_manager.capture_mode = True
    logger.info("▶️ CAPTURE START - User clicked CAPTURE FRAME button - capture_mode = TRUE")
    logger.info(f"Frame manager will now capture frames until STOP is clicked")
    return {"success": True, "mode": "capturing"}


@app.post("/capture-stop")
async def capture_stop():
    """Stop continuous capture mode."""
    frame_manager.capture_mode = False
    logger.info("⏹️ CAPTURE STOP - User clicked STOP CAPTURE button - capture_mode = FALSE")
    logger.info(f"Total frames captured in this session: {frame_manager.detected_objects_count}")
    return {"success": True, "mode": "stopped"}


@app.post("/capture-background")
async def capture_background():
    """Capture single background frame with detection metadata."""
    success = frame_manager.capture_background_frame(detections=frame_manager.current_detections)
    return {"success": success}


@app.get("/stats")
async def get_stats():
    """Get capture statistics."""
    return frame_manager.get_stats()


@app.get("/list-videos")
async def list_videos():
    """List all available videos in the input_dataset folder."""
    videos = list_videos_in_dataset()
    return {
        'success': True,
        'videos': videos,
        'total': len(videos)
    }


@app.post("/process-video")
async def process_video(data: dict = Body(...)):
    """Start processing a video from the dataset."""
    global video_processing_active, video_processing_thread
    
    try:
        video_name = data.get('video_name')
        if not video_name:
            raise HTTPException(status_code=400, detail="video_name is required")
        
        # Get video file path
        videos = list_videos_in_dataset()
        video_path = None
        
        for video in videos:
            if video['name'] == video_name:
                video_path = video['path']
                break
        
        if not video_path:
            raise HTTPException(status_code=404, detail=f"Video not found: {video_name}")
        
        # Start processing thread if not already running
        if video_processing_active:
            return {
                'success': False,
                'message': 'A video is already being processed. Please stop the current video first.'
            }
        
        video_processing_active = True
        video_processing_thread = threading.Thread(
            target=process_video_frames,
            args=(video_path,),
            daemon=True
        )
        video_processing_thread.start()
        
        logger.info(f"✓ Started processing video: {video_name}")
        return {
            'success': True,
            'message': f'Started processing video: {video_name}',
            'video_name': video_name
        }
    
    except Exception as e:
        logger.error(f"Error starting video processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop-video")
async def stop_video():
    """Stop video processing."""
    global video_processing_active
    
    video_processing_active = False
    logger.info("✓ Video processing stopped")
    
    return {
        'success': True,
        'message': 'Video processing stopped'
    }


@app.get("/list-models")
async def api_list_models():
    """Endpoint to list available model versions."""
    models = list_available_models()
    return {
        'success': True,
        'models': models,
        'current_model': MODEL_PATH
    }

@app.post("/switch-model")
async def api_switch_model(data: dict = Body(...)):
    """Endpoint to switch the active model version."""
    model_path = data.get('model_path')
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")
        
    success = frame_manager.switch_model(model_path)
    if success:
        return {'success': True, 'message': f'Switched to model: {model_path}'}
    else:
        return {'success': False, 'message': f'Failed to switch to model: {model_path}'}


@app.get("/live-stream")
async def live_stream():
    """MJPEG stream of current feed."""
    async def frame_generator():
        while True:
            if frame_manager.current_frame is not None:
                # We already drew on the frame in process_frame or process_video_frames
                _, buffer = cv2.imencode('.jpg', frame_manager.current_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.04)  # ~25 FPS

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("  🛡️  SKYGUARD DRONE DETECTION SERVER STARTING")
    logger.info("=" * 70)
    logger.info("")
    logger.info("📍 SERVER CONNECTION DETAILS:")
    logger.info("   🌐 Dashboard (use THIS): http://localhost:7000")
    logger.info("   📤 Frame endpoint:       http://localhost:7000/submit-frame")
    logger.info("")
    logger.info("⚙️  DETECTION SYSTEM:")
    logger.info(f"   🤖 Active model:        {MODEL_PATH}")
    logger.info(f"   🔍 SAHI detection:      {'✓ Enabled' if SAHI_AVAILABLE else '✗ Disabled'}")
    logger.info("")
    logger.info("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=7000)
