# Computer Vision-Based Fatigue Detection System

A real-time facial analysis system that estimates a person's fatigue level from a single image using classical computer vision techniques — Haar Cascade face/eye detection, adaptive thresholding, and morphological image processing. No machine learning model or labeled training data is required; fatigue is scored using a transparent, rule-based algorithm.

## Overview

The system analyzes a facial image and estimates fatigue based on three visual cues:

- **Eye closure** — how "open" the detected eye regions are
- **Eye detection failure** — whether both eyes could even be detected (often indicates squinting/closed eyes)
- **Yawning** — how much of the lower-face region matches an open-mouth pattern

These cues are combined into a single **0–100 fatigue score**, which is then mapped to one of three states: **Normal**, **Tired**, or **Very Tired**.

## How It Works

### 1. Pre-processing
- The input image is converted to grayscale.
- **Histogram equalization** improves contrast under uneven lighting.
- A **Gaussian blur** (5×5) reduces pixel-level noise before detection.

### 2. Face & Eye Detection (Haar Cascades)
- OpenCV's pretrained Haar Cascade classifiers are used to detect:
  - The face region (`haarcascade_frontalface_default.xml`)
  - Eye regions within the face (`haarcascade_eye.xml`)
- If no face is detected, the system reports `"No Face Detected"` and exits early.

### 3. Eye-Closure Analysis
For each detected eye region:
- **Otsu's thresholding** (`THRESH_BINARY_INV + THRESH_OTSU`) automatically separates the eye into foreground/background based on pixel intensity, without a manually-tuned threshold.
- A **morphological opening** operation (3×3 kernel) removes small noise artifacts from the thresholded eye region.
- The **ratio of white pixels** to total eye-region area is computed. A low ratio (`< 0.35`) indicates the eye region shows little contrast/opening — interpreted as a closed or near-closed eye — and adds **+25** to the fatigue score per eye.
- If at least one eye is flagged as closed: **+40** to the score.
- If fewer than 2 eyes are detected at all (often due to squinting or poor detection angle): **+35** to the score.

### 4. Yawning Analysis
- The lower **40% of the face bounding box** is treated as the approximate mouth region.
- The same pipeline (Gaussian blur → Otsu thresholding → morphological closing) is applied to this region.
- The **ratio of detected "open" pixels** to the mouth region's total area is computed. If this ratio exceeds `0.30`, it is interpreted as a wide-open mouth (yawning), adding **+50** to the fatigue score.

### 5. Scoring & Classification
- All contributing scores are summed and capped at **100**.
- The final score maps to a state:

  | Score Range | State |
  |---|---|
  | < 30 | Normal |
  | 30–69 | Tired |
  | ≥ 70 | Very Tired |

### 6. Output
- The detected face is annotated directly on the image with a bounding box and the fatigue state + score (e.g. `"Tired - 65%"`).
- The annotated image is displayed using Matplotlib, alongside a printed summary.

## Tech Stack

- **Python**
- **OpenCV** — Haar Cascade detection, thresholding, morphological operations, image I/O
- **NumPy** — pixel-ratio calculations
- **Matplotlib** — result visualization

## Project Structure

```
.
├── notebook.ipynb     # Full pipeline: preprocessing → detection → scoring → visualization
└── Images/             # Sample input images for testing
```

## Running the Project

1. Install dependencies:
   ```bash
   pip install opencv-python numpy matplotlib
   ```
2. Place a test image in the `Images/` folder.
3. Run:
   ```python
   run_project("Images/your_image.jpg")
   ```
4. The annotated image and fatigue score/state are displayed and printed.

## Design Notes & Limitations

- This is a **single-frame, rule-based** system — it does not track eye state across video frames (e.g. PERCLOS / blink-rate analysis), so it estimates fatigue from one snapshot rather than from sustained behavior over time.
- The scoring weights (+25, +40, +35, +50) and thresholds (0.35, 0.30) were set heuristically based on observed behavior, not learned from labeled fatigue data.
- Haar Cascades are sensitive to head pose, occlusion (e.g. glasses), and lighting; performance degrades on non-frontal faces.
- The mouth region is approximated geometrically (bottom 40% of the face box) rather than detected directly, since no dedicated mouth cascade was used.

## Future Improvements

- Extend to video input and track eye-closure ratio over time (true PERCLOS-based drowsiness detection) instead of single-frame scoring.
- Replace Haar Cascades with a more robust facial landmark detector (e.g. dlib or MediaPipe Face Mesh) for accurate eye/mouth localization regardless of head pose.
- Replace the hand-tuned scoring weights with a small supervised model trained on labeled fatigue/drowsiness datasets.
