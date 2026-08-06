import os

# หา Directory ปัจจุบันของโปรเจกต์
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ตั้งค่ากล้อง ESP32-CAM
ESP32_CAM_IP = "10.117.253.95"  # IP ที่คุณใช้อยู่
ESP32_CAM_CAPTURE_PORT = 80

# ตั้งค่าการประมวลผลวิดีโอ & เว็บ
WEB_HOST = "0.0.0.0"
WEB_PORT = 8000
DETECTION_INTERVAL = 0.15  # ความถี่ในการตรวจจับต่อวินาที

# ตั้งค่า AI Backend ("h5" หรือ "yolo")
DETECTOR_BACKEND = "h5"

# พาธไฟล์โมเดล H5 และ Labels (ใช้ Absolute Path แก้ปัญหา File Not Found)
H5_MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
H5_LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
H5_IMG_SIZE = 224
H5_NORMALIZE = "tm"
H5_CONFIDENCE_THRESHOLD = 0.5  # ปรับลดความมั่นใจลงมาที่ 50% เพื่อให้ตรวจจับติดง่ายขึ้น

# คำค้นหาใน labels.txt
H5_HELMET_LABEL_KEYWORDS = {"helmet", "with", "ใส่หมวก"}
H5_NO_HELMET_LABEL_KEYWORDS = {"no helmet", "without", "ไม่ใส่หมวก"}

# บันทึกข้อมูลและประวัติ
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
VIOLATIONS_LOG_PATH = os.path.join(BASE_DIR, "violations.json")
HISTORY_LIMIT = 50