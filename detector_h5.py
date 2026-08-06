import os
import cv2
import numpy as np
import tf_keras as keras

# 🟢 ใช้ tf_keras (Keras 2) แทน Keras 3 เพื่อรองรับโมเดลจาก Teachable Machine
import tf_keras as keras

class HelmetDetectorH5:
    """
    ตรวจหมวกกันน็อคด้วยโมเดล Keras (.h5) จาก Google Teachable Machine
    รองรับผ่าน tf_keras (Keras v2 compatibility layer)
    """

    def __init__(self, model_path: str, labels_path: str = None, img_size: int = 224,
                 normalize: str = "tm", helmet_keywords=None, no_helmet_keywords=None,
                 conf_threshold: float = 0.5):

        self.img_size = img_size
        self.normalize = normalize
        self.conf_threshold = conf_threshold
        self.helmet_keywords = {k.lower() for k in (helmet_keywords or [])}
        self.no_helmet_keywords = {k.lower() for k in (no_helmet_keywords or [])}

        # 🟢 โหลดโมเดลผ่าน tf_keras
        print(f"[INFO] กำลังโหลดโมเดล H5 ผ่าน tf_keras จาก {model_path}...")
        self.model = keras.models.load_model(model_path, compile=False)
        self.labels = self._load_labels(labels_path)

    def _load_labels(self, labels_path):
        if labels_path and os.path.exists(labels_path):
            labels = {}
            with open(labels_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        labels[int(parts[0])] = parts[1]
                    else:
                        labels[i] = line
            return labels

        n = self.model.output_shape[-1]
        return {i: f"class_{i}" for i in range(n)}

    def _preprocess(self, frame):
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.normalize == "tm":
            img = (img / 127.5) - 1.0  # Normalization สำหรับ Teachable Machine
        else:
            img = img / 255.0
        return np.expand_dims(img, axis=0)

    def detect(self, frame):
        x = self._preprocess(frame)
        preds = self.model.predict(x, verbose=0)
        if isinstance(preds, list):
            preds = preds[0]
        preds = preds[0]

        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = self.labels.get(idx, f"class_{idx}")
        label_lower = label.lower()

        has_helmet = None
        if conf >= self.conf_threshold:
            if any(k in label_lower for k in self.no_helmet_keywords):
                has_helmet = False
            elif any(k in label_lower for k in self.helmet_keywords):
                has_helmet = True

        annotated = frame.copy()
        if has_helmet is True:
            color = (0, 200, 0)
        elif has_helmet is False:
            color = (0, 0, 255)
        else:
            color = (0, 200, 200)

        text = f"{label} ({conf * 100:.1f}%)"
        cv2.rectangle(annotated, (4, 4), (annotated.shape[1] - 4, annotated.shape[0] - 4), color, 4)
        cv2.putText(annotated, text, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        detections = [{"class": label, "confidence": round(conf, 3)}]
        return has_helmet, annotated, detections