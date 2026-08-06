import os
# --- ปิด Warning ของ TensorFlow / Keras ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import json
import time
import random
import datetime
import requests
import threading
import base64
import numpy as np
import tf_keras as keras
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Smart Checkpoint System - Production Ready")

CONFIDENCE_THRESHOLD = 60.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
CAPTURES_DIR = os.path.abspath(os.path.join(BASE_DIR, "evidence"))
DATA_FILE_PATH = os.path.join(BASE_DIR, "history_data.json")

if not os.path.exists(CAPTURES_DIR):
    os.makedirs(CAPTURES_DIR, exist_ok=True)

app.mount("/evidence", StaticFiles(directory=CAPTURES_DIR), name="evidence")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ดึงค่า IP หรือ Public URL จาก Environment Variables (รองรับ Render / Cloud Deployment)
ARDUINO_IP = os.getenv("ARDUINO_IP", "10.117.253.241")
ESP32_CAM_IP = os.getenv("ESP32_CAM_IP", "10.117.253.95")

if ESP32_CAM_IP.startswith("http"):
    ESP32_CAM_CAPTURE_URL = f"{ESP32_CAM_IP.rstrip('/')}/capture"
else:
    ESP32_CAM_CAPTURE_URL = f"http://{ESP32_CAM_IP}/capture"

CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = "C:\\haarcascade_frontalface_default.xml"

if not os.path.exists(CASCADE_PATH):
    print("กำลังดาวน์โหลดไฟล์ Haar Cascade...")
    try:
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        r = requests.get(url, timeout=10)
        with open(CASCADE_PATH, 'wb') as f:
            f.write(r.content)
        print("ดาวน์โหลด Haar Cascade สำเร็จ")
    except Exception as e:
        print(f"ดาวน์โหลดไม่สำเร็จ: {e}")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "helmet_detector_model.h5")

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print("--------------------------------------------------")
        print("โหลดโมเดล Keras และ Face Detector สำเร็จ")
        print("--------------------------------------------------")
    else:
        print("ไม่พบไฟล์โมเดล Keras ในโปรเจกต์")
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")

def load_history():
    if os.path.exists(DATA_FILE_PATH):
        try:
            with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history():
    try:
        with open(DATA_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(violations_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการบันทึกประวัติ: {e}")

violations_history = load_history()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        await websocket.send_json({"type": "history", "data": violations_history})

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

class FastCameraStreamer:
    def __init__(self):
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        session = requests.Session()
        session.headers.update({
            'ngrok-skip-browser-warning': 'true',
            'User-Agent': 'Mozilla/5.0'
        })
        
        while self.running:
            try:
                resp = session.get(ESP32_CAM_CAPTURE_URL, timeout=1.2)
                if resp.status_code == 200:
                    nparr = np.frombuffer(resp.content, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None and frame.size > 0:
                        with self.lock:
                            self.latest_frame = frame
                time.sleep(0.03)
            except Exception:
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

cam_streamer = FastCameraStreamer()

def send_result_to_arduino(state: str):
    try:
        if ARDUINO_IP.startswith("http"):
            url = f"{ARDUINO_IP.rstrip('/')}/state={state}"
        else:
            url = f"http://{ARDUINO_IP}/state={state}"
        requests.get(url, timeout=0.8)
    except Exception:
        pass

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

latest_ai_results = []
is_ai_processing = False
last_auto_record_time = 0

def async_ai_worker(frame_np):
    global latest_ai_results, is_ai_processing
    try:
        h_orig, w_orig = frame_np.shape[:2]
        small_frame = cv2.resize(frame_np, (320, 240))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # ปรับความสว่างและลดแสงสะท้อนรบกวนด้วย CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_equalized = clahe.apply(gray)
        
        # ปรับ Parameter คัดกรองกรอบหลอก/เงามืด
        faces = face_cascade.detectMultiScale(
            gray_equalized, 
            scaleFactor=1.08, 
            minNeighbors=5, 
            minSize=(45, 45)
        )
        
        new_results = []
        scale_x = w_orig / 320.0
        scale_y = h_orig / 240.0

        for (sx, sy, sw, sh) in faces:
            face_box = [int(sx * scale_x), int(sy * scale_y), int((sx + sw) * scale_x), int((sy + sh) * scale_y)]
            
            pad_top = int(sh * scale_y * 0.7)
            pad_side = int(sw * scale_x * 0.2)
            
            y1 = max(0, face_box[1] - pad_top)
            y2 = min(h_orig, face_box[3] + int(sh * scale_y * 0.1))
            x1 = max(0, face_box[0] - pad_side)
            x2 = min(w_orig, face_box[2] + pad_side)

            roi_box = [x1, y1, x2, y2]
            head_roi = frame_np[y1:y2, x1:x2]
            if head_roi.size == 0 or model is None:
                continue

            # กรองภาพมืด หรือภาพที่มีพื้นหลังเรียบเกินไป (Texture/Noise Filter)
            if np.std(head_roi) < 18.0:
                continue

            iou_score = calculate_iou(face_box, roi_box)

            roi_resized = cv2.resize(head_roi, (224, 224))
            roi_normalized = (roi_resized.astype("float32") / 127.5) - 1.0
            roi_input = np.expand_dims(roi_normalized, axis=0)

            preds = model.predict(roi_input, verbose=0)[0]

            if len(preds) > 1:
                pred_class = np.argmax(preds)
                conf = float(preds[pred_class]) * 100.0
                is_wearing = (pred_class == 0)
            else:
                pred_val = float(preds[0])
                is_wearing = pred_val < 0.5
                conf = (1.0 - pred_val) * 100.0 if is_wearing else pred_val * 100.0

            overall_score = conf * max(iou_score, 0.6)
            new_results.append((face_box[0], face_box[1], face_box[2], face_box[3], is_wearing, conf, overall_score))

        latest_ai_results = new_results
    except Exception as e:
        print(f"AI Worker Error: {e}")
    finally:
        is_ai_processing = False

def generate_cam_feed():
    global is_ai_processing, last_auto_record_time
    frame_counter = 0

    while True:
        frame_np = cam_streamer.get_frame()
        if frame_np is not None:
            frame_counter += 1
            
            if frame_counter % 2 == 0 and not is_ai_processing:
                is_ai_processing = True
                threading.Thread(target=async_ai_worker, args=(frame_np.copy(),), daemon=True).start()

            annotated_frame = frame_np.copy()
            for (x1, y1, x2, y2, is_wearing, conf, overall_score) in latest_ai_results:
                color = (0, 255, 0) if is_wearing else (0, 0, 255)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                status_text = "Helmet" if is_wearing else "No Helmet"
                text_display = f"{status_text} ({overall_score:.1f}%)"

                cv2.putText(annotated_frame, text_display, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if not is_wearing and conf >= CONFIDENCE_THRESHOLD:
                    current_time = time.time()
                    if current_time - last_auto_record_time >= 3.0:
                        last_auto_record_time = current_time
                        
                        now = datetime.datetime.now()
                        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
                        filename = f"violation_{timestamp_str}.jpg"
                        
                        filepath = os.path.join(CAPTURES_DIR, filename)
                        cv2.imwrite(filepath, annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                        _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                        b64_str = base64.b64encode(buffer).decode('utf-8')
                        image_data_url = f"data:image/jpeg;base64,{b64_str}"

                        student_id = f"STD{random.randint(10000, 99999)}"

                        new_record = {
                            "student_id": student_id,
                            "date": now.strftime("%Y-%m-%d"),
                            "time": now.strftime("%H:%M:%S"),
                            "image_url": image_data_url,
                            "is_wearing": False,
                            "status_text": "ไม่สวมหมวก",
                            "score": f"{overall_score:.1f}%"
                        }
                        
                        violations_history.insert(0, new_record)
                        save_history()
                        
                        import asyncio
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(manager.broadcast({"type": "new_violation", "data": new_record}))
                            loop.close()
                        except Exception:
                            pass
                        
                        threading.Thread(target=send_result_to_arduino, args=("NO_HELMET",)).start()

            _, encoded = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n')
            
            time.sleep(0.01)
        else:
            blank_img = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_img, "Connecting Camera...", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            _, encoded = cv2.imencode('.jpg', blank_img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n')
            time.sleep(0.1)

@app.get("/", response_class=HTMLResponse)
def serve_index():
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse(content="<h3>ไม่พบ index.html</h3>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_cam_feed(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/trigger_pir")
def trigger_pir():
    print("ได้รับสัญญาณตรวจจับวัตถุจาก PIR Sensor")
    return JSONResponse(content={"status": "success", "message": "PIR Triggered"})

@app.websocket("/ws/live-status")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

@app.delete("/api/clear-history")
async def clear_history():
    violations_history.clear()
    save_history()
    await manager.broadcast({"type": "history", "data": []})
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    # ปรับพอร์ตไดนามิกตามสภาพแวดล้อม Render / Local
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)