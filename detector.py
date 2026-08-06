from ultralytics import YOLO


class HelmetDetector:
    """ห่อโมเดล YOLOv8 ไว้ตรวจว่าในเฟรมมีคนใส่หมวกกันน็อคหรือไม่"""

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 helmet_names=None, no_helmet_names=None):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.helmet_names = {n.lower() for n in (helmet_names or [])}
        self.no_helmet_names = {n.lower() for n in (no_helmet_names or [])}

    def detect(self, frame):
        """
        คืนค่า (has_helmet, annotated_frame, detections)
        has_helmet:
          True  = เจอคนใส่หมวก
          False = เจอคนไม่ใส่หมวก
          None  = ตรวจไม่พบคน/หมวกเลยในเฟรม (โมเดลไม่มั่นใจ หรือไม่มีใครอยู่หน้ากล้อง)
        """
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        r = results[0]
        annotated = r.plot()  # เฟรมที่วาดกรอบ + label ให้อัตโนมัติ

        detections = []
        has_helmet = None
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id].lower()
            conf = float(box.conf[0])
            detections.append({"class": cls_name, "confidence": round(conf, 3)})

            if cls_name in self.no_helmet_names:
                has_helmet = False
            elif cls_name in self.helmet_names and has_helmet is None:
                has_helmet = True

        return has_helmet, annotated, detections
