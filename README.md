# 🤖 Action-Engagement-Collaboration Triad (AEC-Triad)

[![HRI 2026](https://img.shields.io/badge/HRI-2026-4A90E2?style=flat-square)](https://dl.acm.org/doi/abs/10.1145/3757279.3788807)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Code Available](https://img.shields.io/badge/Code-Available-brightgreen?style=flat-square)](.)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-orange?style=flat-square)](https://huggingface.co/datasets/ArvindSihag/M3_Multimodal_Human_Robot_Collaboration_Dataset)
[![Paper](https://img.shields.io/badge/📄-Paper-blue?style=flat-square)](https://dl.acm.org/doi/abs/10.1145/3757279.3788807)

> **The Action-Engagement-Collaboration Triad: A Multimodal Analytical Framework for Human-Robot Collaboration**  
> *HRI 2026* | [Read Paper](https://dl.acm.org/doi/abs/10.1145/3757279.3788807)

---

## 📋 Table of Contents
- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [📊 Dataset Access](#-dataset-access)
- [🚀 Quick Start](#-quick-start)
- [📁 Repository Structure](#-repository-structure)
- [📖 Citation](#-citation)
- [🙌 Acknowledgements](#-acknowledgements)
- [📬 Contact](#-contact)

---

## 🎯 Overview

The **Action-Engagement-Collaboration (AEC) Triad** is a novel framework that integrates multimodal signals to analyze and evaluate human-robot collaboration in industrial assembly environments. This repository contains the official implementation of our HRI 2026 paper.

Our framework provides a unified analytical approach across three behavioral scales:

| Layer | Focus | Key Metrics |
|-------|-------|-------------|
| **MICRO** | Action-level behavior | Action entropy, Diversity ratio |
| **MESO** | Engagement dynamics | Engagement stability, High engagement ratio |
| **MACRO** | Collaboration performance | Collaboration score, Interaction efficiency |

---

## ✨ Features

- 🎥 **Multimodal Analysis** - Third-person video, top-view RGB/D, and IMU sensor data
- 🤝 **Collaboration Metrics** - Action, Engagement, and Collaboration scores
- 📊 **Interactive Visualization** - Real-time monitoring of collaboration quality
- 📈 **Benchmarking Tools** - Standardized evaluation on HRC datasets
- 🔬 **Fine-grained Annotations** - 280 action primitives across 17 categories
- 📍 **Engagement Labels** - Frame-wise classification (HIGHLY ENGAGED to DISENGAGED)

---

## 📊 Dataset Access

### M³ Multimodal Human-Robot Collaboration Dataset

We provide a comprehensive multimodal dataset for human-robot collaboration research, available on Hugging Face.

| Dataset Details | |
|-----------------|-|
| **Dataset Name** | M³ Multimodal Human-Robot Collaboration Dataset |
| **Platform** | 🤗 Hugging Face |
| **Size** | 4.31 GB |
| **Subjects** | 8 Operators |
| **Modalities** | Video (C1, C2), Depth, IMU |
| **Annotations** | 280 Actions, Engagement Labels |
| **License** | CC BY-NC 4.0 |
| **Status** | 🔒 Gated Access (Approval Required) |

### 🔐 How to Access

**Step 1: Visit the Dataset Page**  
🔗 [ArvindSihag/M3_Multimodal_Human_Robot_Collaboration_Dataset](https://huggingface.co/datasets/ArvindSihag/M3_Multimodal_Human_Robot_Collaboration_Dataset)

**Step 2: Request Access**  
Click the **"Request Access"** button and fill out the form with your:
- Full Name
- Institution/Organization
- Position/Role
- Research Area
- Purpose of Use
- Expected Duration of Use

**Step 3: Approval & Download**  
- You'll receive an email notification once approved
- Download the required zip files
- Extract using: `unzip filename.zip`

### 📦 Dataset Files

| File Name | Size |
|-----------|------|
| 2024-9-20_11-40-54-563.zip | 728 MB |
| 2024-9-20_12-37-54-780.zip | 346 MB |
| 2024-9-20_18-26-57-871.zip | 1.16 GB |
| 2024-9-21_16-55-29-534.zip | 2.07 GB |

### 🗺️ Operator ID Mapping

| Operator ID | Raw Directory Name |
|-------------|-------------------|
| ID-1 | 2024-9-21_16-55-29-534 |
| ID-2 | 2024-9-21_18-02-33-428 |
| ID-3 | 2024-9-25_16-51-01-958 |
| ID-4 | 2024-9-27_18-00-07-834 |
| ID-5 | 2024-12-02_16-50-19-914 |
| ID-6 | 2024-9-25_18-21-19-101 |
| ID-7 | 2024-9-26_17-59-47-126 |
| ID-8 | 2024-10-05_17-23-20-277 |

### 📁 Dataset Structure

```bash
2024-9-20_11-40-54-563/          # Each directory as operator ID
│
├── c1_clips/                    # Third-person videos (primitive segmented)
│   └── *.mp4                    # Video files
│
├── c2_rgb_clips/                # Top-view RGB videos
│   └── *.mp4                    # RGB video files
│
├── c2_depth_clips/              # Depth streams
│   └── *.npy                    # Depth data files
│
├── imu_data/                    # IMU sensor CSV files
│   ├── left_hand_imu.csv        # Left hand IMU data
│   └── right_hand_imu.csv       # Right hand IMU data
│
├── annotations/                 # Action + engagement labels
│   ├── actions.csv              # Action primitive annotations
│   └── engagement.csv           # Frame-wise engagement labels
│
├── metrics/                     # MICRO / MESO / MACRO Layer outputs
│   ├── micro_metrics.csv        # MICRO layer analysis results
│   ├── meso_metrics.csv         # MESO layer analysis results
│   └── macro_metrics.csv        # MACRO layer analysis results
│
└── README.md                    # Dataset documentation
```


### 📜 License & Usage Terms

This dataset is released under **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)**:

- ✔ Research use allowed
- ✔ Attribution required
- ❌ Commercial use prohibited

**Usage Policy:**
- Dataset access requires approval
- For research purposes only
- Redistribution is not allowed
- Proper citation is mandatory

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/arvindsihag/m3_analyzer.git
cd m3_analyzer
pip install -r requirements.txt
pip install -e .
```
### Basic Usage
```bash

from m3_analyzer import AECTriad
from m3_analyzer import load_data

analyzer = AECTriad()

data = load_data(
    video_path="c1_clips/video.mp4",
    depth_path="c2_depth_clips/depth.npy",
    imu_path="imu_data/sensor.csv",
    annotations_path="annotations/actions.csv"
)

micro_metrics = analyzer.compute_micro(data.actions)
meso_metrics = analyzer.compute_meso(data.engagement)
macro_metrics = analyzer.compute_macro(data)

analyzer.visualize(
    micro_metrics=micro_metrics,
    meso_metrics=meso_metrics,
    macro_metrics=macro_metrics,
    save_path="results/visualization.png"
)
```

### Loading from Hugging Face
```bash
from datasets import load_dataset

# Login first: huggingface-cli login
dataset = load_dataset(
    "ArvindSihag/M3_Multimodal_Human_Robot_Collaboration_Dataset",
    split="train"
)
```
### 📖 Citation
```bash
@inproceedings{arvind2026action,
  title={The Action-Engagement-Collaboration Triad: A Multimodal Analytical Framework for Human-Robot Collaboration},
  author={Arvind and Mehta, Naval Kishore and Kumar, Himanshu and Saurav, Sumeet and Singh, Sanjay},
  booktitle={Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction (HRI)},
  pages={1293--1297},
  year={2026},
  doi={10.1145/3757279.3788807}
}
```
### 🙌 Acknowledgements
```bash
Developed at:
CSIR-Central Electronics Engineering Research Institute (CEERI), Pilani
Academy of Scientific and Innovative Research (AcSIR), India
```
### 📖 Citation
```bash
📬 Contact
Arvind Sihag
📧 arvind.ceeri24a@acsir.res.in
🔗 GitHub | Hugging Face
```
### 📖 Citation
```bash
⭐ Star Us!
If you find this work useful, please give us a star ⭐ and cite our paper!
```


