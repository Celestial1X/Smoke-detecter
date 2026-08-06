import json
import os


def load_students(path: str) -> dict:
    """
    โหลดฐานข้อมูล ทะเบียนรถ -> นักเรียน จากไฟล์ JSON
    รูปแบบไฟล์ (ดูตัวอย่างใน students.json):
    {
      "กข1234กรุงเทพมหานคร": {"student_id": "6301234", "name": "สมชาย ใจดี"}
    }
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_student(plate_text: str, db: dict):
    """หา record นักเรียนจากข้อความป้ายทะเบียนที่ OCR อ่านมา (เทียบแบบตัดช่องว่างออกก่อน)"""
    normalized = plate_text.replace(" ", "")
    for plate, info in db.items():
        if plate.replace(" ", "") == normalized:
            return info
    return None
