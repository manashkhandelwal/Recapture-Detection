# Screen Recapture Detector

This repository contains a fast, classical signal-based pipeline to detect whether an image is a real photo or a photo of a screen (recapture). 

It operates completely on-device using OpenCV, NumPy, and SciPy. It requires no deep learning model weights, no GPU, and no API calls, making it extremely lightweight and cost-effective.

## Live Demo
Check out the live Streamlit application here: 
👉 **[Insert Streamlit App Link Here]**

## Installation & Requirements

Ensure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install numpy opencv-python scipy
```
*(Note: If you are running the Streamlit app locally, you will also need to `pip install streamlit`)*

## Usage (CLI)

You can run predictions on any image using the `predict.py` script. The script will automatically load the optimal signal weights from `weights.json`. 

### 1. Basic Score Prediction
Outputs a probability score from 0.0 to 1.0 (where 0.0 = definitely a real photo, 1.0 = definitely a screen):
```bash
python predict.py path/to/image.jpg
```

### 2. Predict Label
Outputs a clear `REAL` or `SCREEN` classification based on the calibrated threshold:
```bash
python predict.py path/to/image.jpg --label
```

### 3. Detailed Debug Output
Prints a full per-signal breakdown, individual feature contributions, latency (in ms), and the confidence score. Great for understanding *why* the model made its decision:
```bash
python predict.py path/to/image.jpg --debug
```

### 4. JSON Output
Outputs the full detailed results as a formatted JSON object, which is useful for piping into other tools or APIs:
```bash
python predict.py path/to/image.jpg --json
```

## Custom Weights
If you retrain the model on a custom dataset using `train.py`, you can pass your custom weights file:
```bash
python predict.py path/to/image.jpg --weights custom_weights.json
```
