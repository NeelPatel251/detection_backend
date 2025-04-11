# backend/main.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import base64
import time
from typing import List
import asyncio
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load YOLO model and DeepSort tracker
try:
    logger.info("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    names = model.names
    logger.info("YOLO model loaded successfully")
    
    logger.info("Initializing DeepSort tracker...")
    tracker = DeepSort(max_age=30)
    logger.info("DeepSort tracker initialized")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New connection established. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Connection closed. Remaining connections: {len(self.active_connections)}")

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

def process_frame(frame, fps):
    try:
        # Run YOLO detection
        results = model(source=frame, conf=0.5, iou=0.5)[0]
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for result in results.boxes:
                coords = result.xyxy[0].cpu().numpy().tolist()
                conf = float(result.conf[0])
                class_id = int(result.cls[0])
                label = names[class_id]
                x1, y1, x2, y2 = map(int, coords)
                w, h = x2 - x1, y2 - y1
                if w > 0 and h > 0:
                    detections.append(([x1, y1, w, h], conf, label))
        
        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # Draw bounding boxes and labels
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            label = track.det_class
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}, {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add FPS counter
        # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        # Return original frame if processing fails
        return frame

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info("WebSocket connection established")
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            
            if data.startswith('data:image/jpeg;base64,'):
                try:
                    # Extract the base64 encoded image
                    img_data = data.replace('data:image/jpeg;base64,', '')
                    img_bytes = base64.b64decode(img_data)
                    
                    # Convert to numpy array
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        logger.warning("Received invalid image data")
                        continue
                    
                    # Process the frame
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:  # Avoid division by zero
                        fps = frame_count / elapsed_time
                    else:
                        fps = 0
                    
                    processed_frame = process_frame(frame, fps)
                    
                    # Convert back to base64 for sending
                    _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    img_encoded = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send_text(f"data:image/jpeg;base64,{img_encoded}")
                    
                    # Reset FPS counter periodically to get current FPS
                    if frame_count > 100:
                        frame_count = 0
                        start_time = time.time()
                        
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    # Send error message to client
                    await websocket.send_text(f"error:Image processing failed: {str(e)}")
            elif data == "stop":
                logger.info("Received stop command")
                break
            else:
                logger.warning(f"Received unknown data: {data[:30]}...")
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in manager.active_connections:
            manager.disconnect(websocket)

@app.get("/")
async def root():
    return {"message": "Object Detection API is running. Connect to /ws for real-time detection."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)