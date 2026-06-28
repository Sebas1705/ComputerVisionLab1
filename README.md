# Practica1VisionArtificial

Computer Vision university lab — road sign detector using MSER and HSV colour masking.

Detects rectangular traffic sign panels in images using a classical computer vision pipeline:
MSER region detection → grouping → aspect-ratio filtering → HSV colour masking →
contrast normalisation and perspective correction.

---

## Pipeline

```
Input images
    │
    ▼
a_gray_before/      Grayscale conversion
    │
    ▼
b_gray_after/       Histogram equalisation (contrast improvement)
    │
    ▼
c_regioned/         MSER region detection (raw bounding boxes)
    │
    ▼
d_groupped_regioned/ cv2.groupRectangles clustering
    │
    ▼
e_filter_regioned/  Aspect-ratio & size filter (remove noise)
    │
    ▼
f_cropped/          Crop + resize candidate regions (160×80 px)
    │
    ▼
g_cropped_mask/     HSV blue-channel mask (keep sign panels, reject background)
    │
    ▼
h_final_regioned/   Final bounding boxes drawn on original images
    │
    ▼
i_final_cropped/    Final cropped sign panels
    │
    ▼
j_improve_images/   CLAHE contrast + Hough perspective correction
```

---

## Project Structure

```
Practica1VisionArtificial/
├── proyect/
│   ├── src/
│   │   ├── classes/
│   │   │   ├── Detector.py       # MSER detection, grouping, filtering, cropping, HSV masking
│   │   │   ├── Normalizer.py     # CLAHE contrast + Hough perspective correction
│   │   │   └── Tester.py         # Orchestrates the full pipeline
│   │   ├── common/
│   │   │   └── FileFuncs.py      # Image I/O helpers
│   │   ├── settings.py           # All tunable constants (MSER params, CLAHE, Canny, Hough)
│   │   └── main.py               # Entry point
│   ├── images/                   # Intermediate results (a_gray_before → j_improve_images)
│   ├── files/                    # Output text files (bounding box coordinates)
│   └── .gitignore
├── imagenesTest/                 # Raw input test images (102 images)
├── imagenesResultado/            # Reference result images
├── notebook_PracticaObligatoria1.ipynb   # Jupyter exploration notebook
└── README.md
```

---

## Running

```bash
cd proyect/src
python main.py
```

Reads images from `proyect/images/test_selected/`, runs the full detection pipeline, and saves
all intermediate results to the `proyect/images/` subdirectories.

---

## Key Parameters (`settings.py`)

| Parameter | Value | Description |
|---|---|---|
| `DELTA` | 4 | MSER stability threshold |
| `MIN_AREA` / `MAX_AREA` | 1000 / 80000 | MSER region size bounds |
| `MIN_RATIO` / `MAX_RATIO` | 0.4 / 4 | Aspect ratio filter for candidate regions |
| `CROPPED_TAM` | (160, 80) | Resize target for cropped candidates |
| `CLIP_LIMIT` | 2.0 | CLAHE clip limit |
| `THREASHOLD1/2` | 50 / 150 | Canny edge detection thresholds |

---

## Tech Stack

- Python 3.12
- OpenCV (`cv2`) — MSER, CLAHE, Canny, Hough, HSV masking
- NumPy

---

## Course Context

Lab 1 of the *Computer Vision* (Visión Artificial) course.
Goal: implement a classical (non-DL) road-sign detection pipeline and evaluate detection
accuracy against ground-truth bounding box annotations.
