# Gesture-Controlled Mouse with Intel oneAPI

A real-time hand gesture recognition system that controls your mouse cursor, optimized with Intel oneAPI for cross-architecture acceleration.

## 🔧 Key Features
- 👆 Index finger tracking for cursor movement
- 🤏 Pinch gesture for mouse clicks
- ⚡ Intel-optimized performance via:
  - OpenVINO toolkit for AI inference
  - oneDNN acceleration for MediaPipe
  - Intel IPP in OpenCV
- 🖼️ Air drawing mode with gesture-based color switching

## 🚀 Intel oneAPI Integration
This project leverages multiple Intel technologies:

| Component | Intel Tech Used | Benefit |
|-----------|-----------------|---------|
| **AI Inference** | OpenVINO Toolkit | 3-5x faster inference on Intel CPUs/GPUs |
| **Math Operations** | oneDNN | Accelerated landmark calculations |
| **Image Processing** | Intel IPP (via OpenCV) | Optimized frame processing |
| **Thread Management** | Intel TBB | Efficient multi-threading |

## 📦 Installation

### Prerequisites
- Intel CPU (6th Gen+) or GPU (Iris Xe/Arc)
- Python 3.9+
- Windows/Linux

```bash
# Clone the repository
git clone https://github.com/devznsh/AirCursor.git
cd AirCursor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies (Intel-optimized versions)
pip install -r requirements.txt
```
## 🏗️ Project Structure
```

gesture-mouse-oneapi/
├── models/                   # OpenVINO IR models
├── gesture_mouse.py          # Main control script
├── air_drawing.py           # Drawing mode
├── openvino_utils.py        # Intel acceleration helpers
└── requirements.txt         # Intel-optimized packages

```


