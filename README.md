<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=FFD400,000000&height=200&section=header&text=SmartClass%20AI&fontSize=60&fontColor=FFD400&fontAlignY=35&desc=Classroom%20Intelligence%20Console&descColor=B3B3B3&descSize=20&descAlignY=55" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10-FFD400?style=for-the-badge&logo=python&logoColor=000000"/>
  <img src="https://img.shields.io/badge/Streamlit-1.38-FFD400?style=for-the-badge&logo=streamlit&logoColor=000000"/>
  <img src="https://img.shields.io/badge/YOLOv8-Face_Detection-FFD400?style=for-the-badge&logo=github&logoColor=000000"/>
  <img src="https://img.shields.io/badge/MediaPipe-0.10.9-FFD400?style=for-the-badge&logo=google&logoColor=000000"/>
  <img src="https://img.shields.io/badge/Gemini-1.5_Flash-FFD400?style=for-the-badge&logo=google&logoColor=000000"/>
</p>

<p>
  <img src="https://img.shields.io/badge/OpenCV-4.9-000000?style=for-the-badge&labelColor=FFD400&color=111111"/>
  <img src="https://img.shields.io/badge/ONNX_Runtime-1.20-000000?style=for-the-badge&labelColor=FFD400&color=111111"/>
  <img src="https://img.shields.io/badge/RandomForest-Focus_Model-000000?style=for-the-badge&labelColor=FFD400&color=111111"/>
  <img src="https://img.shields.io/badge/License-MIT-000000?style=for-the-badge&labelColor=FFD400&color=111111"/>
  <img src="https://img.shields.io/badge/Status-Active-000000?style=for-the-badge&labelColor=3ED67F&color=111111"/>
</p>

<br/>

> **"Teachers can't monitor 30 students at once. AI can."**

<br/>

</div>

---

## ◈ What Is SmartClass AI?

**SmartClass AI** is a complete **Classroom Intelligence Console** — an AI-powered system that monitors student attention in real time, detects emotions, scores individual engagement, fires smart behavioral alerts, and generates professional teaching reports powered by **Google Gemini 1.5 Flash**.

The system uses **YOLOv8** for robust multi-face detection in real classroom photos — detecting small, angled, and partially visible faces that traditional detectors miss.

```
Input                    AI Pipeline                         Output
──────────────────────────────────────────────────────────────────────
Photo / Video    →   YOLOv8 Face Detection          →   Focus Score
Webcam Snapshot  →   OpenCV Cascade (fallback)      →   Emotion Tags
Class Recording  →   MediaPipe Face Mesh (fallback) →   Agent Alerts
                 →   Eye Landmark Analysis           →   AI Reports
                 →   RandomForest Classifier         →   Teacher Dashboard
                 →   CNN Emotion Model (ONNX)        →   Class Summary
                 →   Google Gemini 1.5 Flash         →
```

---

## ◈ The Problem

| Challenge | Impact |
|-----------|--------|
| Teachers cannot watch 30+ students simultaneously | Students fall behind unnoticed |
| No real-time attention data | Disengagement goes undetected |
| Standard face detectors fail on classroom photos | Small/angled faces missed |
| No AI-powered teaching effectiveness reports | Missed improvement opportunities |

**SmartClass AI solves all of this.**

---

## ◈ Key Features

```
◆ DETECTION ENGINE (3-Layer Fallback Chain)
  ├── YOLOv8 face detection — primary (best for classrooms)
  ├── OpenCV Cascade + CLAHE enhancement — secondary fallback
  ├── MediaPipe whole-image scan — tertiary fallback
  ├── Focus scoring 0–100 using eye landmark analysis
  ├── Head pose estimation — looking forward or away
  ├── Blink rate monitoring
  └── Multi-face detection — up to 30+ students simultaneously

◆ INPUT MODES
  ├── 📸 Photo Upload       — single or group classroom photos
  ├── 🎥 Video Analysis     — frame-by-frame recording analysis
  └── 📷 Live Snapshot      — webcam-based real-time capture

◆ BEHAVIORAL INTELLIGENCE (5 Agent Rules)
  ├── Rule 1 — Low sustained engagement → immediate alert
  ├── Rule 2 — Repeated distraction → student flagged
  ├── Rule 3 — Not looking at board → attention warning
  ├── Rule 4 — Class average drops → teacher notification
  └── Rule 5 — Majority distracted → critical class alert

◆ AI REPORTS (Gemini 1.5 Flash)
  ├── 📋 Today's Class Summary
  ├── 👤 Students Who Need Improvement
  └── 📊 Teaching Effectiveness Report (rated /10)

◆ TEACHER DASHBOARD (8 Pages)
  ├── Per-student engagement grid (color-coded)
  ├── Class overview — Focused / Moderate / Distracted counts
  ├── Session analytics — 4 chart tabs
  ├── Smart alert log with severity levels
  └── CSV export for all session data
```

---

## ◈ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | Streamlit 1.38 | Web dashboard |
| **Primary Detection** | YOLOv8 (ultralytics) | Robust multi-face detection |
| **Secondary Detection** | OpenCV 4.9 + CLAHE | Enhanced cascade fallback |
| **Tertiary Detection** | MediaPipe 0.10.9 | Whole-image face scan |
| **Face Landmarks** | MediaPipe Face Mesh | 468 3D landmarks per face |
| **Emotion Model** | CNN → ONNX Runtime 1.20 | 7-class emotion classification |
| **Focus Model** | RandomForest (sklearn) | Focused / Distracted prediction |
| **AI Reports** | Google Gemini 1.5 Flash | Natural language teaching reports |
| **Charts** | Plotly | Interactive session analytics |
| **Data** | Pandas, NumPy 1.26.4 | Session logging & processing |
| **Training Data** | FER2013 | Emotion model dataset |

---

## ◈ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│           Photo │ Video │ Webcam Snapshot                       │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│               3-LAYER FACE DETECTION CHAIN                      │
│                                                                 │
│  Layer 1: YOLOv8  ──→  Found?  ──→  Use results               │
│                             ↓ No                               │
│  Layer 2: OpenCV + CLAHE Enhancement ──→  Found?  ──→  Use    │
│                                               ↓ No            │
│  Layer 3: MediaPipe full-image scan ──→  Use results          │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│           PER-FACE ANALYSIS (each detected face)               │
│                                                                 │
│   Face crop + padding                                           │
│         ↓                          ↓                           │
│   MediaPipe landmarks        CNN Emotion Model                  │
│   (EAR, head pose, gaze)     (ONNX Runtime)                    │
│         ↓                          ↓                           │
│   RandomForest Classifier    7 Emotion Classes                  │
│   → Focus score 0–100        → Confidence score                │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                   BEHAVIORAL AGENT                              │
│     5 rules → Critical / High / Medium severity alerts         │
│     Student flag count → Chronic distraction detection          │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              GOOGLE GEMINI 1.5 FLASH                            │
│    Analyzes session data → 3 professional report types          │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│               TEACHER DASHBOARD (Streamlit)                     │
│   8 pages │ Real-time grid │ Charts │ Alerts │ CSV Export       │
└─────────────────────────────────────────────────────────────────┘
```

---

## ◈ Project Structure

```
SmartClassroom/
│
├── app.py                        ← Main Streamlit app (8 pages)
│
├── modules/
│   ├── __init__.py
│   ├── emotion_detector.py       ← CNN inference via ONNX
│   ├── engagement_scorer.py      ← EAR + head pose + gaze scoring
│   ├── face_tracker.py           ← YOLOv8 + fallback multi-face detection
│   ├── agent.py                  ← Rule-based behavioral intelligence
│   └── report_generator.py       ← Gemini AI report generation
│
├── utils/
│   ├── __init__.py
│   └── logger.py                 ← CSV session logging
│
├── models/
│   ├── emotion_model.onnx        ← Trained CNN emotion model
│   ├── focus_model.pkl           ← Trained RandomForest focus model
│   └── yolov8n-face.pt           ← YOLOv8 face detection (auto-downloaded)
│
├── data/
│   └── sessions.csv              ← Auto-generated session history
│
├── .env                          ← API keys (never commit)
├── .gitignore
├── requirements.txt
└── packages.txt
```

---

## ◈ Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 **Dashboard** | Live overview — photo, video, or snapshot analysis |
| 👥 **Class Analysis** | Multi-student batch upload |
| 📷 **Single Student** | Deep-dive analysis — photo or video |
| 🏫 **Classroom View** | YOLOv8 multi-face detection in classroom photo |
| 📈 **Analytics** | 4-tab charts — engagement, emotions, states, per-student |
| 🤖 **AI Report** | 3 Gemini-powered professional reports |
| 🔔 **Alerts** | Agent alert log with severity levels |
| 📁 **History** | Full session data + CSV export |
| ℹ **About** | Project details, tech stack, architecture |

---

## ◈ Installation

### Prerequisites

- Python 3.10+
- pip
- Google Gemini API key → [Get free key](https://aistudio.google.com/app/apikey)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Hamna-Munir/SmartClassroom-AI.git
cd SmartClassroom-AI

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
echo GEMINI_API_KEY=your_key_here > .env

# 5. Run the app
streamlit run app.py
```

> YOLOv8 face model (`yolov8n-face.pt`) downloads automatically on first run.

App opens at → `http://localhost:8501`

---

## ◈ Requirements

```txt
streamlit==1.38.0
opencv-python==4.9.0.80
mediapipe==0.10.9
numpy==1.26.4
pandas==2.2.2
plotly==5.22.0
scikit-learn==1.7.2
google-genai==2.8.0
python-dotenv==1.2.2
Pillow==10.4.0
onnxruntime==1.20.1
ultralytics==8.2.0
```

---

## ◈ How To Use

### Classroom View — Multi-Face Detection
```
1. Go to 🏫 Classroom View
2. Upload any classroom photo
3. YOLOv8 detects ALL faces automatically
4. Each student gets: focus score, state, eye contact status
5. Per-student grid shows 🟢 Focused / 🟡 Moderate / 🔴 Distracted
6. Agent fires alerts automatically
```

### Single Student Analysis
```
1. Go to 📷 Single Student
2. Choose: Photo Upload / Video Upload / Live Snapshot
3. Enter student name (optional)
4. View: Focus Score, Emotion Breakdown, Score Breakdown
```

### AI Teaching Report
```
1. Analyze some students first
2. Go to 🤖 AI Report
3. Choose report type:
   - 📋 Today's Class Summary
   - 👤 Students Who Need Improvement
   - 📊 Teaching Effectiveness Report
4. Click Generate → Gemini writes your report
```

---

## ◈ YOLOv8 Face Detection

SmartClass AI uses **YOLOv8** as the primary face detector — specifically designed for classroom scenarios where standard detectors fail.

```
Why YOLOv8?
────────────────────────────────────────────────────
✅ Detects small faces (students far from camera)
✅ Handles angled/side-view faces
✅ Works with partially occluded faces
✅ Robust in varying lighting conditions
✅ Processes 30+ faces in one image
✅ Fast inference — suitable for real-time use

Detection Chain:
────────────────────────────────────────────────────
1. YOLOv8 (conf=0.25, iou=0.4)   ← Primary
2. OpenCV Cascade + CLAHE          ← Fallback 1
3. MediaPipe full-image scan       ← Fallback 2

Image Enhancement (before cascade):
────────────────────────────────────────────────────
• Auto-upscale if image < 800px
• CLAHE contrast enhancement
• Sharpening filter
• Face crop padding (+15%)
```

---

## ◈ AI Models

### Emotion Detection Model
```
Architecture:   CNN (Convolutional Neural Network)
Dataset:        FER2013 (35,887 labeled images)
Format:         ONNX (cross-platform, no TensorFlow needed)
Accuracy:       66.6%
Output:         7 emotion classes
Classes:        Happy · Sad · Angry · Neutral · Fear · Surprise · Disgust
Input:          48×48 grayscale face crop
```

### Focus Detection Model
```
Algorithm:      RandomForest Classifier
Classes:        Focused (1) / Distracted (0)
Features:       6 eye landmark measurements
  ├── Left EAR  (Eye Aspect Ratio)
  ├── Right EAR
  ├── Average EAR
  ├── Left eye width
  ├── Right eye width
  └── Inter-eye distance
Output:         Focus probability → 0–100 score
Adjustments:    Head pose penalty + emotion weight
```

---

## ◈ Engagement Scoring Formula

```
Score = (EAR_score × 0.40 + Pose_score × 0.35 + Gaze_score × 0.25)
        × Emotion_weight − Blink_penalty

Where:
  EAR_score     = eye openness ratio (0–100)
  Pose_score    = 100 if looking forward, penalized by tilt angle
  Gaze_score    = 90 if iris centered, else 40
  Emotion_weight:
    Happy    = 1.15  (boosts score)
    Neutral  = 1.00
    Sad      = 0.65  (reduces score)
    Angry    = 0.45
  Blink_penalty = min(10, blink_count)

States:
  ≥ 60  →  🟢 Focused
  35–59 →  🟡 Moderate
  < 35  →  🔴 Distracted
```

---

## ◈ Agent Rules

```
Rule 1  LOW_ENGAGEMENT       Score < 35% for 3+ readings      → HIGH alert
Rule 2  PATTERN_DETECTED     Student flagged 3+ times          → CRITICAL alert
Rule 3  NOT_ATTENDING        Not looking + score < 40%         → MEDIUM alert
Rule 4  CLASS_ATTENTION_DROP Class avg < 45% (3 readings)     → HIGH alert
Rule 5  MAJORITY_DISTRACTED  50%+ students distracted          → CRITICAL alert
```

---

## ◈ Roadmap

- [x] Single student focus + emotion analysis
- [x] YOLOv8 multi-face classroom detection
- [x] 3-layer detection fallback chain
- [x] Smart behavioral agent (5 rules)
- [x] 3 AI report types (Gemini)
- [x] Video frame analysis
- [x] Session analytics dashboard (4 tabs)
- [x] Image enhancement pipeline (CLAHE + sharpen)
- [ ] Phase 2 — Real-time video stream
- [ ] Phase 2 — Student ID tracking across frames
- [ ] Phase 2 — Cloud deployment
- [ ] Phase 3 — SaaS teacher portal

---

## ◈ Developer

<div align="center">

<br/>

**Hamna Munir**

*AI/ML Engineer · Software Engineering Student*

[![GitHub](https://img.shields.io/badge/GitHub-Hamna--Munir-FFD400?style=for-the-badge&logo=github&logoColor=000000)](https://github.com/Hamna-Munir)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Hamna27-FFD400?style=for-the-badge&logo=huggingface&logoColor=000000)](https://huggingface.co/Hamna27)

<br/>

*Building AI systems that solve real-world problems in education.*

<br/>

</div>

---

## ◈ License

```
MIT License — free to use, modify, and distribute with attribution.
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=FFD400,000000&height=120&section=footer&text=SmartClass%20AI&fontSize=24&fontColor=FFD400&fontAlignY=65"/>

**SmartClass AI · Classroom Intelligence Console**

*Built with ❤️ by Hamna Munir · 2026*

</div>

