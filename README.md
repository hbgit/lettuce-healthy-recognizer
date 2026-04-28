# Lettuce Disease Detection with YOLO11 and RT-DETR

This repository features a Deep Learning-based computer vision system designed to identify and classify health conditions in lettuce crops. The project compares the performance of the latest **YOLO11** architecture against the Real-Time Detection Transformer (**RT-DETR**) to provide a robust solution for precision agriculture.

## 📌 Project Overview

Early identification of leaf diseases is vital for crop management. This project implements an automated pipeline for:
- **Dataset Acquisition:** Integration with Roboflow for seamless data loading.
- **Data Engineering:** Automated rebalancing of training and testing sets (80/20 split) to ensure model generalization.
- **Model Training:** Comparative analysis between **YOLO11l** and **RT-DETR-l**.
- **Inference & Validation:** Real-time processing of high-resolution images and videos from local vegetable gardens.

## 🛠️ Technology Stack

- **Framework:** [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO11 & RT-DETR)
- **Dataset Management:** Roboflow API
- **Language:** Python 3.12+
- **Deep Learning:** PyTorch with CUDA acceleration (NVIDIA RTX 4050 tested)
- **Computer Vision:** OpenCV, Matplotlib

## 📂 Dataset Details

The dataset (`disease-lettuce-h69zb`) contains images classified into several categories, including:
- **Healthy**
- **Septoria Blight**
- **Downy Mildew**
- **Bacterial**
- **Lettuce Mosaic Virus**
- **Lettuce Anthracnose**

The images are pre-processed to a resolution of **640x640** (for training) and **1280x1280** (for high-precision inference).

## 🚀 Performance Comparison

| Model | Task | Image Size | Batch Size | Hardware |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO11l** | Detection | 640 | 4 | RTX 4050 |
| **RT-DETR-l** | Detection | 640 | 4 | RTX 4050 |

### Key Results
The models were validated on real-world videos (e.g., `horta-elvis.MP4`), demonstrating high frames-per-second (FPS) during inference while maintaining high mAP (mean Average Precision) across multiple disease classes.

## ⚙️ Installation & Usage

1. **Install Dependencies:**
   ```bash
   pip install ultralytics roboflow opencv-python torch matplotlib
   ```

2. **Download Dataset:**
   Configure your Roboflow API key in the notebook and run the data acquisition cell to download the `disease-lettuce` project.

3. **Train Models:**
   The training script is configured to automatically detect CUDA and start the training battery for both YOLO and RT-DETR architectures.

4. **Run Inference:**
   ```python
   from ultralytics import YOLO
   model = YOLO('best.pt')
   results = model.predict(source='your_video.mp4', save=True, imgsz=1280)
   ```

## 📊 Evaluation
The project includes visualization scripts to compare **Ground Truth** labels against **Model Predictions**, allowing for a qualitative analysis of the detector's performance in varied lighting and backgrounds.

---

### Author
Developed as part of an Artificial Intelligence project focused on agricultural technology.
