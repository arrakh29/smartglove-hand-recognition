# SmartGlove Hand Recognition

Real-time hand posture and gesture recognition using multisensory smart gloves, Hybrid GCN-MLP, and Spatio-Temporal Graph Convolutional Networks (ST-GCN).

---

## Overview

This project presents a dual-pipeline framework for real-time hand posture and gesture recognition using **Rokoko Smart Gloves**. The system combines graph-based spatial learning and spatio-temporal motion modelling to classify both static hand postures and dynamic hand gestures from wearable sensor data streamed via UDP.

The framework is divided into two parallel subsystems:

- **Posture Recognition** — uses a Hybrid GCN-MLP architecture to classify instantaneous hand configurations from joint-angle and geometric features.
- **Gesture Recognition** — uses ST-GCN with temporal feature fusion to model full motion trajectories across time.

Both subsystems run concurrently during live inference.

---

## Results

| Task | Samples | Classes | Accuracy | Macro F1 |
|------|---------|---------|----------|----------|
| Posture Recognition | 2,700 | 6 | 98.61% | 98.61% |
| Gesture Recognition | 385 | 7 | 100.00% | 100.00% |

---

## Dataset

### Posture Classes (6)
`inferior-pincer` · `palmar` · `pincer` · `radial-digital` · `radial-palmar` · `rake`

### Gesture Classes (7)
`up` · `down` · `left` · `right` · `rotate_clockwise` · `rotate_counterclockwise` · `idle`

Data were collected in a controlled laboratory environment with multiple subjects and repetitions. Posture data are stored in **CSV** format (statistical window features); gesture data are stored in **NPZ** format (skeleton sequences, temporal features, labels).

---

## System Pipeline

### Posture Recognition
```
UDP Stream → Joint-Angle Extraction → Geometric Feature Extraction
          → Statistical Window Aggregation → Hybrid GCN-MLP → Posture Class
```

### Gesture Recognition
```
UDP Stream → Coordinate Normalisation → Sequence Resampling (T=30)
          → Temporal Feature Extraction → ST-GCN Fusion → Gesture Class
```

---

## Architecture

### Hybrid GCN-MLP (Posture)
- **GCN branch** — models joint-angle features as a graph with anatomically defined edges
- **MLP branch** — processes 30 discriminative geometric features as a global vector
- **Fusion** — concatenated outputs fed to a softmax classifier (6 classes)

### ST-GCN with Temporal Feature Fusion (Gesture)
- **ST-GCN branch** — spatio-temporal graph convolution over 21-joint skeleton sequences
- **Temporal Feature Branch** — 1D CNN over 44-dimensional per-frame feature vectors
- **Fusion Classifier** — combined prediction over 7 gesture classes

---

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Scikit-learn
- OpenCV
- Matplotlib
- Rokoko Studio (for glove streaming)
- CUDA-enabled GPU (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Hardware

| Component | Specification |
|-----------|--------------|
| Sensor | Rokoko Smart Gloves (IMU + EMF) |
| Processor | Intel Core i7 (11th Gen) |
| RAM | 16 GB |
| GPU | NVIDIA GeForce RTX 3060 |
| OS | Windows 11 |

---

## Usage

### Recording Mode
Captures 3-second raw joint-coordinate sequences for dataset construction.

```bash
python main.py --mode record --label <class_name>
```

### Inference Mode
Runs both posture and gesture recognition in real time from live UDP stream.

```bash
python main.py --mode infer
```

UDP stream is expected at `127.0.0.1:14043` in Rokoko JSON packet format.

---

## Demo

### Real-Time Inference

<p align="center">
  <a href="https://www.youtube.com/watch?v=abc123xyz">
    <img src="https://img.youtube.com/vi/abc123xyz/maxresdefault.jpg" width="700">
  </a>
</p>

<p align="center">
  ▶ Click the image to watch the demo video
</p>
