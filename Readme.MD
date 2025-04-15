# ASL DETR Project

Train a DETR Transformer model to detect American Sign Language (ASL) letters (A-Z) using YOLOv8 annotations.

## Setup

1. Download the ASL Letters Dataset from Roboflow in YOLOv8 format.
2. Place it in `dataset/` with `train/`, `valid/`, and `data.yaml`.
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
