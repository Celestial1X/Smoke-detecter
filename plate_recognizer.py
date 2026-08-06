import os
import re


def clean_plate_text(raw: str) -> str:
    """ตัดอักขระที่ไม่ใช่ตัวเลข/ตัวอักษรไทย-อังกฤษออก (ช่วยกรอง noise จาก OCR)"""
    return re.sub(r"[^0-9ก-๙A-Za-z]", "", raw)


class PlateRecognizer:
    """
    ตรวจจับป้ายทะเบียน + อ่านตัวอักษรด้วย OCR (ANPR)

    ถ้ามีไฟล์โมเดล YOLO สำหรับตรวจจับป้ายทะเบียนโดยเฉพาะ (PLATE_MODEL_PATH) จะใช้ตัวนั้นหาตำแหน่ง
    ป้ายก่อนแล้วค่อย crop ไป OCR (แม่นกว่า)
    ถ้าไม่มีไฟล์โมเดล จะ fallback ไป OCR ทั้งเฟรมเลย (ใช้งานได้ทันทีแต่แม่นน้อยกว่า)
    """

    def __init__(self, plate_model_path: str, conf_threshold: float = 0.4,
                 ocr_langs=("th", "en")):
        self.conf_threshold = conf_threshold
        self.model = None

        if plate_model_path and os.path.exists(plate_model_path):
            from ultralytics import YOLO
            self.model = YOLO(plate_model_path)

        import easyocr
        self.reader = easyocr.Reader(list(ocr_langs), gpu=False)

    def read_plates(self, frame):
        """คืนค่า list ของ {bbox, text, confidence}"""
        if self.model is not None:
            return self._read_with_detector(frame)
        return self._read_whole_frame(frame)

    def _read_with_detector(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        plates = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            text = self._ocr(crop)
            if text:
                plates.append({
                    "bbox": (x1, y1, x2, y2),
                    "text": text,
                    "confidence": round(float(box.conf[0]), 3),
                })
        return plates

    def _read_whole_frame(self, frame):
        text = self._ocr(frame)
        if not text:
            return []
        h, w = frame.shape[:2]
        return [{"bbox": (0, 0, w, h), "text": text, "confidence": 0.0}]

    def _ocr(self, crop):
        results = self.reader.readtext(crop)
        if not results:
            return ""
        # เรียงจากบนลงล่าง เพราะป้ายทะเบียนไทยมักมี 2 บรรทัด (เลขทะเบียน + จังหวัด)
        results.sort(key=lambda r: r[0][0][1])
        text = " ".join(clean_plate_text(r[1]) for r in results if clean_plate_text(r[1]))
        return text.strip()
