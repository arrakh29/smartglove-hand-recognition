````md
# SmartGlove Hand Recognition

Real-time hand posture and gesture recognition framework using multisensory smart gloves, Hybrid GCN-MLP, and Spatio-Temporal Graph Convolutional Networks (ST-GCN).

---

## Overview

This project presents a dual-pipeline framework for real-time hand posture and gesture recognition using Rokoko Smart Gloves. The system combines graph-based spatial learning and spatio-temporal motion modelling to classify both static hand postures and dynamic hand gestures from wearable sensor data.

The framework consists of:

### Posture Recognition Subsystem
- Hybrid GCN-MLP architecture
- Joint-angle and geometric feature extraction
- Statistical window aggregation

### Gesture Recognition Subsystem
- ST-GCN with temporal feature fusion
- Sequence normalisation and temporal resampling
- Real-time motion trajectory modelling

The system supports continuous real-time inference through UDP streaming from Rokoko Smart Gloves.

---

## Features

- Real-time hand posture recognition
- Real-time dynamic gesture recognition
- Dual-pipeline architecture
- Graph-based skeletal feature learning
- Temporal motion modelling
- UDP-based smart glove streaming
- Statistical feature aggregation
- ST-GCN temporal fusion framework

---

## Dataset

### Hand Posture Classes
- inferior-pincer
- palmar
- pincer
- radial-digital
- radial-palmar
- rake

### Hand Gesture Classes
- up
- down
- left
- right
- rotate_clockwise
- rotate_counterclockwise
- idle

---

## System Architecture

### Posture Recognition Pipeline
1. Joint coordinate acquisition
2. Joint-angle extraction
3. Geometric feature extraction
4. Statistical window aggregation
5. Hybrid GCN-MLP classification

### Gesture Recognition Pipeline
1. Skeleton sequence acquisition
2. Coordinate normalisation
3. Sequence resampling
4. Temporal feature extraction
5. ST-GCN temporal fusion classification

---

## Experimental Results

| Task | Accuracy | Macro F1-score |
|------|----------|----------------|
| Posture Recognition | 98.61% | 98.61% |
| Gesture Recognition | 100.00% | 100.00% |

---

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Scikit-learn
- OpenCV
- Matplotlib
- Rokoko Studio
- CUDA-enabled GPU (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## Project Structure

```text
smartglove-hand-recognition/
│
├── dataset/
├── models/
├── posture/
├── gesture/
├── realtime/
├── utils/
├── figures/
├── checkpoints/
├── requirements.txt
└── README.md
```

---

## Real-Time Streaming

The system receives skeletal data from Rokoko Smart Gloves through UDP streaming:

```text
127.0.0.1:14043
```

Make sure Rokoko Studio streaming is enabled before running inference.

---

## Running the Project

### Train Posture Model

```bash
python train_posture.py
```

### Train Gesture Model

```bash
python train_gesture.py
```

### Run Real-Time Inference

```bash
python realtime_inference.py
```

---

## Citation

If you use this project in your research, please cite:

```bibtex
@inproceedings{smartglove2026,
  title={Hand Posture and Gesture Recognition Using Dual-Branch GCN on Smart Gloves},
  author={Anonymous},
  year={2026}
}
```

---

## Acknowledgment

The authors would like to thank the institution and laboratory staff for providing facilities, technical assistance, and continuous support throughout this research and experimental development process.

---

## License

This project is intended for academic and research purposes.

```
```
