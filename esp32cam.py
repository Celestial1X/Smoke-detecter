import threading
import time
import urllib.request

import cv2
import numpy as np


class ESP32CamStream:
    """
    อ่าน MJPEG stream จาก ESP32-CAM ที่รันเฟิร์มแวร์ตัวอย่างมาตรฐาน (CameraWebServer)
    สตรีมอยู่ที่ http://<ip>:81/stream ตามค่า default ของเฟิร์มแวร์นั้น
    """

    def __init__(self, ip: str, stream_port: int = 81, capture_port: int = 80):
        self.stream_url = f"http://{ip}:{stream_port}/stream"
        self.capture_url = f"http://{ip}:{capture_port}/capture"
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.connected = False
        self.thread = None
        self.error = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return True

    def _reader(self):
        while self.running:
            try:
                resp = urllib.request.urlopen(self.stream_url, timeout=5)
                self.connected = True
                self.error = None
                buf = b""
                while self.running:
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    buf += chunk
                    start = buf.find(b"\xff\xd8")  # JPEG SOI marker
                    end = buf.find(b"\xff\xd9")    # JPEG EOI marker
                    if start != -1 and end != -1 and end > start:
                        jpg = buf[start:end + 2]
                        buf = buf[end + 2:]
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            with self.lock:
                                self.frame = img
                self.connected = False
            except Exception as e:
                self.connected = False
                self.error = f"เชื่อมต่อ ESP32-CAM ({self.stream_url}) ไม่ได้: {e}"
                time.sleep(2)  # รอแล้วลองต่อใหม่

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
