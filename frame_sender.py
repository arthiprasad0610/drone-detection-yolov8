"""
frame_sender.py
===============
Frame Sender Client for Live Server

PURPOSE:
    Sends video frames from your video extraction/processing program
    to the Advanced Live Video Streaming Server.
    
    This acts as a bridge between your video processing pipeline
    and the interactive live streaming server.

USAGE:
    # Method 1: Extract frames from a video file
    python frame_sender.py --video "path/to/video.mp4" --fps 30
    
    # Method 2: From webcam
    python frame_sender.py --webcam
    
    # Method 3: Use in your Python code
    from frame_sender import FrameSender
    
    sender = FrameSender()
    frame = cv2.imread("frame.jpg")
    sender.send_frame(frame)

EXAMPLE (Integration with your video extraction):
    import cv2
    from frame_sender import FrameSender
    
    sender = FrameSender(server_url="http://localhost:7000")
    
    # Your video processing loop
    cap = cv2.VideoCapture("data/raw/video.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Your processing here...
        # processed_frame = your_detection_pipeline(frame)
        
        # Send to live server
        success = sender.send_frame(frame)
        if success:
            print(f"Frame sent: {sender.frame_count}")

REQUIREMENTS:
    pip install opencv-python requests
"""

import cv2
import requests
import argparse
import logging
import time
from pathlib import Path
import io

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class FrameSender:
    """Sends frames to the Live Streaming Server."""
    
    def __init__(self, server_url: str = "http://localhost:7000"):
        self.server_url = server_url.rstrip('/')
        self.frame_endpoint = f"{self.server_url}/submit-frame"
        self.frame_count = 0
        
        # Test connection
        self.test_connection()
    
    def test_connection(self):
        """Test if server is running."""
        try:
            response = requests.get(f"{self.server_url}/stats", timeout=2)
            if response.status_code == 200:
                logger.info(f"✓ Connected to server: {self.server_url}")
                return True
        except requests.exceptions.ConnectionError:
            logger.error(f"✗ Cannot connect to server at {self.server_url}")
            logger.error("Make sure live_server_advanced.py is running")
            return False
    
    def send_frame(self, frame) -> bool:
        """
        Send a frame to the server.
        
        Args:
            frame: numpy array (OpenCV format)
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        try:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Send to server
            files = {'frame': frame_bytes}
            response = requests.post(
                self.frame_endpoint,
                files=files,
                timeout=5
            )
            
            if response.status_code == 200:
                self.frame_count += 1
                data = response.json()
                
                # Log detection info
                if data.get('has_object'):
                    logger.info(f"Frame {self.frame_count}: ✓ Object detected "
                              f"({data.get('detections_count', 0)} detections)")
                
                return True
            else:
                logger.error(f"Server returned {response.status_code}")
                return False
        
        except requests.exceptions.Timeout:
            logger.error("Frame send timeout")
            return False
        except Exception as e:
            logger.error(f"Error sending frame: {e}")
            return False
    
    def send_video_file(self, video_path: str, fps: int = 30):
        """
        Send all frames from a video file.
        
        Args:
            video_path: Path to video file
            fps: Frames per second to send (default 30)
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return
        
        logger.info(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video info: {total_frames} frames @ {video_fps} FPS")
        logger.info(f"Sending at: {fps} FPS")
        
        frame_delay = 1.0 / fps
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                
                # Send frame
                success = self.send_frame(frame)
                
                # Progress indicator
                if frame_num % 30 == 0:
                    progress = (frame_num / total_frames) * 100
                    logger.info(f"Progress: {frame_num}/{total_frames} "
                              f"({progress:.1f}%) - Sent: {self.frame_count}")
                
                # Control frame rate
                time.sleep(frame_delay)
        
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        finally:
            cap.release()
            logger.info(f"✓ Complete! Sent {self.frame_count} frames")
    
    def send_webcam(self, fps: int = 30):
        """
        Send frames from webcam.
        
        Args:
            fps: Frames per second (default 30)
        """
        logger.info("Opening webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
        
        frame_delay = 1.0 / fps
        
        logger.info("Sending frames from webcam (press Ctrl+C to stop)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Send frame
                success = self.send_frame(frame)
                
                if self.frame_count % 30 == 0:
                    logger.info(f"Frames sent: {self.frame_count}")
                
                # Control frame rate
                time.sleep(frame_delay)
        
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        finally:
            cap.release()
            logger.info(f"✓ Complete! Sent {self.frame_count} frames")


# ==============================================================
# COMMAND LINE INTERFACE
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Send video frames to Live Streaming Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python frame_sender.py --video "path/to/video.mp4"
  python frame_sender.py --video "path/to/video.mp4" --fps 15
  python frame_sender.py --webcam
  python frame_sender.py --server http://192.168.1.100:7000 --video video.mp4
        """
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to video file to send'
    )
    
    parser.add_argument(
        '--webcam',
        action='store_true',
        help='Use webcam as source'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second (default: 30)'
    )
    
    parser.add_argument(
        '--server',
        type=str,
        default='http://localhost:7000',
        help='Server URL (default: http://localhost:7000)'
    )
    
    args = parser.parse_args()
    
    # Create sender
    sender = FrameSender(server_url=args.server)
    
    # Send frames
    if args.video:
        sender.send_video_file(args.video, fps=args.fps)
    elif args.webcam:
        sender.send_webcam(fps=args.fps)
    else:
        parser.print_help()
