import threading
import time

import cv2


class Camera:
    """อ่านภาพจากกล้อง USB แบบ background thread เก็บเฟรมล่าสุดไว้ให้ดึงไปใช้ได้ตลอดเวลา"""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.thread = None
        self.error = None

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            self.error = "เปิดกล้องไม่ได้ — เช็คว่าเสียบกล้องอยู่ และไม่มีโปรแกรมอื่นเปิดใช้งานกล้องนี้ค้างไว้"
            return False

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()
