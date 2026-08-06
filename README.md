# Smart Checkpoint — กล้องขึ้นเว็บ + AI (.h5) ตรวจหมวก + ANPR

เว็บแอป (FastAPI) พร้อมใช้งานครบวงจร:
- ดึงภาพสดจาก **ESP32-CAM** (`http://192.168.1.187:81/stream`) ขึ้นเว็บตลอดเวลา
- รัน **AI โมเดล .h5 ที่เทรนไว้แล้ว** ตรวจทุกเฟรมแบบ real-time (ไม่ต้องรอ PIR ก่อน) แล้ววาดผลทับ
  บนวิดีโอที่โชว์บนเว็บเลย
- คู่ขนานกัน: poll PIR จาก Arduino UNO R4 WiFi → สั่งจอ LED Matrix (✕/○) → ถ้าไม่ใส่หมวก
  อ่านป้ายทะเบียน (ANPR) + เทียบฐานข้อมูลนักเรียน + บันทึกภาพหลักฐาน + ขึ้นแถวใหม่ในตาราง log
  แบบเรียลไทม์ผ่าน WebSocket

## สิ่งที่ต้องทำก่อนรัน (สำคัญ)

### 1. เอาไฟล์โมเดลมาวางในโฟลเดอร์โปรเจกต์
วางไฟล์ `.h5` ที่เทรนไว้แล้ว (และ `labels.txt` ถ้ามี — ถ้าเทรนผ่าน Google Teachable Machine จะได้
มาคู่กันเป็น `keras_model.h5` + `labels.txt` อยู่แล้ว) ไว้ในโฟลเดอร์นี้ แล้วแก้ `config.py`:

```python
H5_MODEL_PATH = "helmet_model.h5"   # แก้เป็นชื่อไฟล์จริง
H5_LABELS_PATH = "labels.txt"
```

### 2. เช็ค input shape / label ให้ตรงกับที่ตั้งไว้
รันสคริปต์ตรวจสอบที่แถมมาให้ก่อนสตาร์ทเว็บ:

```bash
python inspect_h5_model.py helmet_model.h5
```

จะพิมพ์ **input shape** (ปรับ `H5_IMG_SIZE` ให้ตรง เช่น 224x224 → `H5_IMG_SIZE = 224`) และ
**output shape** (จำนวนคลาส ต้องตรงกับจำนวนบรรทัดใน `labels.txt`)

### 3. เช็คว่า label ตรงกับ keyword ที่ตั้งไว้
เปิด `labels.txt` ดูว่าเขียนชื่อคลาสว่าอะไร แล้วแก้ให้ `H5_HELMET_LABEL_KEYWORDS` /
`H5_NO_HELMET_LABEL_KEYWORDS` ใน `config.py` มีคำที่ตรงกับชื่อคลาสจริง (ตัวพิมพ์เล็ก-ใหญ่ไม่สำคัญ
เพราะระบบแปลงเป็นตัวเล็กให้ก่อนเทียบ) เช่นถ้า `labels.txt` เขียนว่า

```
0 Wearing Helmet
1 Not Wearing Helmet
```

ให้ตั้ง `H5_HELMET_LABEL_KEYWORDS = {"wearing helmet"}` และ
`H5_NO_HELMET_LABEL_KEYWORDS = {"not wearing"}`

### 4. ตั้งค่า IP ของอุปกรณ์
```python
ARDUINO_IP = "..."     # จาก Serial Monitor ตอน Arduino boot
ESP32_CAM_IP = "192.168.1.187"   # ตั้งไว้แล้ว
```

## ติดตั้งและรัน

```bash
cd helmet-checkpoint-web
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main.py
```

เปิด `http://localhost:8000` — จะเห็นภาพสดจาก ESP32-CAM พร้อมกรอบ + label ผลตรวจจาก AI ทับอยู่
บนวิดีโอตลอดเวลา และแถบสถานะด้านบน (ESP32-CAM / Arduino / AI Model / ผลตรวจล่าสุด) อัพเดตทุก 3 วิ

> **หมายเหตุการติดตั้ง:** `tensorflow` และ `easyocr` (ใช้ PyTorch ข้างใต้) เป็นไลบรารีขนาดใหญ่
> ครั้งแรกที่ `pip install` และรันจะใช้เวลาสักพัก และ `easyocr` ต้องมีอินเทอร์เน็ตตอนรันครั้งแรก
> เพื่อโหลดโมเดลภาษา

## โครงสร้างไฟล์

```
helmet-checkpoint-web/
├── main.py               # FastAPI app หลัก: 2 background loop + video feed + websocket + status API
├── config.py               # ตั้งค่าทั้งหมด (IP, โมเดล, ANPR, ฯลฯ) — จุดที่ต้องแก้เป็นหลัก
├── detector_h5.py            # ตรวจหมวกด้วยโมเดล .h5 (ค่า default ของโปรเจกต์)
├── detector.py                 # ตรวจหมวกด้วยโมเดล YOLOv8 (.pt) — ใช้เมื่อสลับ DETECTOR_BACKEND="yolo"
├── inspect_h5_model.py           # สคริปต์เช็ค input/output shape ของไฟล์ .h5 ก่อนตั้งค่า
├── arduino_client.py               # เรียก /motion-status, /helmet-ok, /no-helmet, /idle บน Arduino
├── esp32cam.py                       # อ่าน MJPEG stream จาก ESP32-CAM (:81/stream)
├── plate_recognizer.py                 # ตรวจจับป้ายทะเบียน + OCR (ANPR)
├── student_db.py                         # lookup ทะเบียนรถ -> รหัสนักเรียน จาก students.json
├── students.json                           # ฐานข้อมูลตัวอย่าง (แก้ไข/เพิ่มเองได้)
├── labels.txt                                # ตัวอย่าง label file (แก้ให้ตรงกับโมเดลจริง)
├── violations_log.json                         # ไฟล์เก็บ log อัตโนมัติ (สร้างตอนรันครั้งแรก)
├── requirements.txt
├── templates/index.html                          # หน้าแดชบอร์ด (สถานะระบบ + กล้องสด + log)
└── static/evidence/                                # ภาพหลักฐานที่บันทึกอัตโนมัติ
```

## ระบบทำงานยังไง

```
                        ┌── LOOP 1: video_loop() ──────────────────────────────┐
                        │  ดึงเฟรมจาก ESP32-CAM ทุก 0.3 วิ (ตลอดเวลา)             │
                        │  รันโมเดล .h5 → วาดกรอบ+label ทับเฟรม                  │
                        │  → อัพเดตสิ่งที่โชว์บน /video_feed                       │
                        │  → เก็บผลล่าสุด (has_helmet) ไว้ใช้ต่อใน loop 2          │
                        └───────────────────────────────────────────────────────┘

                        ┌── LOOP 2: arduino_loop() ─────────────────────────────┐
                        │  poll /motion-status จาก Arduino ทุก 0.5 วิ             │
                        │  ▼ มีความเคลื่อนไหว                                     │
                        │  เอาผลล่าสุดจาก loop 1 (ถ้าใหม่ไม่เกิน 2 วิ) มาตัดสิน:      │
                        │    ใส่หมวก    → GET /helmet-ok  (LED Matrix โชว์ ○)      │
                        │    ไม่ใส่หมวก → GET /no-helmet  (LED Matrix โชว์ ✕)      │
                        │                → อ่านป้ายทะเบียน (ANPR)                  │
                        │                → เทียบ students.json หา student_id      │
                        │                → เซฟภาพหลักฐาน + เพิ่มแถว log             │
                        │                → broadcast ผ่าน WebSocket ไปหน้าเว็บ      │
                        │  ▼ นิ่งเกิน 5 วิ → GET /idle (จอดับ)                     │
                        └───────────────────────────────────────────────────────┘
```

- `VIOLATION_COOLDOWN` (default 8 วิ) กันไม่ให้บันทึกซ้ำรัวๆ ตอนคนเดิมยืนอยู่หน้ากล้องนาน
- log ทั้งหมดอยู่ในไฟล์ `violations_log.json` — ลบ/แก้ไฟล์นี้ได้ตรงๆ ถ้าต้องการเคลียร์ประวัติ
- ถ้า ESP32-CAM หรือ Arduino หลุด แอปยังรันได้ปกติ แค่จุดสถานะบนเว็บจะเปลี่ยนเป็นสีแดง/เทา —
  เช็ค terminal ตอนรัน `python main.py` จะมี `[warn]` บอกสาเหตุเสมอ

## หมายเหตุเรื่อง ANPR (อ่านป้ายทะเบียน)

- ถ้ายังไม่มีโมเดลตรวจจับตำแหน่งป้ายทะเบียนโดยเฉพาะ (`PLATE_MODEL_PATH` ไม่มีไฟล์อยู่จริง) ระบบจะ
  OCR ทั้งเฟรมแทน ใช้งานได้แต่แม่นน้อยกว่า
- ถ้า OCR อ่านทะเบียนได้แต่หาไม่เจอใน `students.json` จะขึ้น `รหัสนักเรียน: ไม่ทราบ` — เพิ่มรายการ
  เข้า `students.json` แล้วครั้งต่อไปจะจับคู่ได้เอง

## อยากสลับกลับไปใช้โมเดล YOLO (.pt) แทน .h5

แก้บรรทัดเดียวใน `config.py`:
```python
DETECTOR_BACKEND = "yolo"
```
แล้วตั้งค่า `YOLO_MODEL_PATH` / `HELMET_CLASS_NAMES` / `NO_HELMET_CLASS_NAMES` ให้ตรงกับโมเดลนั้น
