"""
สคริปต์เช็คข้อมูลโมเดล .h5 ก่อนเอาไปตั้งค่าใน config.py

วิธีใช้:
    python inspect_h5_model.py path/to/model.h5

จะพิมพ์ input shape (เอาไปตั้ง H5_IMG_SIZE) และ output shape (จำนวนคลาส)
ให้ตรงกับที่ตั้งไว้ใน config.py และ labels.txt
"""
import sys

import tensorflow as tf

path = sys.argv[1] if len(sys.argv) > 1 else "helmet_model.h5"

model = tf.keras.models.load_model(path, compile=False)
model.summary()

print("\n=== สรุปสำหรับตั้งค่าใน config.py ===")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)
print(f"-> จำนวนคลาสของโมเดลนี้คือ {model.output_shape[-1]}")
print("-> ถ้า input shape เป็น (None, 224, 224, 3) ให้ตั้ง H5_IMG_SIZE = 224 ใน config.py")
