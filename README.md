# 🚗 Tunisian Vehicle License Plate Detection & Recognition  
### AI Challenge – Bounding Box Detection + OCR

![License Plate Banner](https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1400&q=60)

> Building an end-to-end AI system to detect and recognize Tunisian vehicle license plates for intelligent traffic monitoring.

---

## 🔗 Competition Link

Official Zindi Competition Page:  
https://zindi.africa/competitions/artificial-intelligence-challenge-advanced

---

## 📌 Overview

This AI challenge focuses on developing a complete **Automatic License Plate Recognition (ALPR)** system for Tunisian vehicles.

Participants are provided with:

- 📸 **900 annotated car images** (bounding boxes for plates)
- 🔤 **900 plate text samples** (for OCR training)

The goal is to build:

1. 🎯 A robust license plate detection model (bounding box localization)
2. 🔎 An accurate OCR model to recognize plate characters

The **Top 5 teams (by December 7)** advance to the finals, where solutions are evaluated for real-world deployment in traffic camera monitoring systems.

---

## 🎯 Objectives

- Detect Tunisian vehicle license plates from car images
- Accurately extract plate numbers using OCR
- Ensure generalization across lighting, angle, and motion conditions
- Develop models suitable for traffic surveillance systems

---

## 🧠 Technical Approach

This project is divided into two main tasks:

---

## 1️⃣ License Plate Detection (Object Detection)

We use object detection models to localize license plates.

### Possible Models

- YOLOv8
- Faster R-CNN
- EfficientDet
- SSD

Example (YOLO-based training):

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=50, imgsz=640)
```

### Evaluation Metrics

- mAP (mean Average Precision)
- IoU (Intersection over Union)
- Precision / Recall

---

## 2️⃣ Optical Character Recognition (OCR)

After detecting plates, we crop them and perform text recognition.

### OCR Approaches

- CRNN (Convolutional Recurrent Neural Network)
- Tesseract (baseline)
- Transformer-based OCR
- EasyOCR / PaddleOCR fine-tuning

Example (EasyOCR inference):

```python
import easyocr

reader = easyocr.Reader(['ar','en'])
result = reader.readtext("plate_crop.jpg")
```

---

## 📂 Project Structure

```
tunisian-license-plate-ai/
│
├── data/
│   ├── images/
│   ├── annotations/
│
├── detection/
│   ├── train_detection.py
│
├── ocr/
│   ├── train_ocr.py
│
├── inference.py
├── requirements.txt
├── license_plates_detection_and_recogintion.ipynb
└── README.md
```

---

## 🔄 End-to-End Pipeline

1. Input vehicle image  
2. Detect license plate (bounding box)  
3. Crop detected region  
4. Apply OCR model  
5. Output structured plate number  

Example inference flow:

```python
detections = detection_model(image)
plate_crop = crop_plate(image, detections)
plate_text = ocr_model(plate_crop)
```

---

## 📊 Evaluation Criteria

- Detection mAP
- OCR Character Accuracy
- Full Plate Recognition Accuracy
- End-to-End Accuracy
- Robustness under real-world conditions

---

## 🚦 Real-World Application

Final solutions may be deployed in:

- 🚥 Traffic camera monitoring systems
- 🚔 Law enforcement applications
- 🅿️ Smart parking systems
- 🛣️ Toll collection systems

The system must handle:

- Motion blur
- Low-light conditions
- Partial occlusion
- Various camera angles

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/tunisian-license-plate-ai.git
cd tunisian-license-plate-ai
pip install -r requirements.txt
```

---

## ▶️ Training

### Train Detection Model

```bash
python detection/train_detection.py
```

### Train OCR Model

```bash
python ocr/train_ocr.py
```

---

## 🤖 Inference

```bash
python inference.py --image test.jpg
```

---

## 🏆 Competition Timeline

- Model Development Phase
- Leaderboard Ranking
- 📅 Top 5 Teams Selected by December 7
- Final Evaluation & Deployment Review

---

## 📈 Future Improvements

- Multi-camera tracking integration
- Real-time edge deployment (Jetson / Edge TPU)
- Model quantization for faster inference
- Arabic-specific OCR fine-tuning
- Video stream processing

---

## 📄 License

MIT License

---

## ⭐ Support

If you found this project helpful, please ⭐ star the repository!
