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
├── project/
│   ├── src/
│   │   ├── classes/
│   │   │   ├── Detector.py       # MSER detection, grouping, filtering, cropping, HSV masking
│   │   │   ├── Normalizer.py     # CLAHE contrast + Hough perspective correction
│   │   │   └── Tester.py         # Orchestrates the full pipeline
│   │   ├── common/
│   │   │   └── FileFuncs.py      # Image I/O helpers
│   │   ├── settings.py           # All tunable constants (MSER params, CLAHE, Canny, Hough)
│   │   └── main.py               # Entry point
│   ├── images/
│   │   ├── test_selected/        # Sample input images (8 of the 20 selected test images)
│   │   └── a_gray_before/ … j_improve_images/   # Pipeline outputs (empty, regenerated on run)
│   ├── files/                    # Ground-truth annotations, detections and IoU comparison
│   └── .gitignore
├── imagenesResultado/            # Output folder for notebook_PracticaObligatoria1.ipynb
├── notebook_PracticaObligatoria1.ipynb   # Jupyter exploration notebook
├── p1_test0.ipynb                # Experiments (HSV masking, perspective correction)
└── README.md
```

---

## Dataset

The input images are road scenes taken from a car, provided by the course
(*Visión Artificial*, Computer Engineering, Universidad Rey Juan Carlos, 2023/24) as its
test image collection. That collection is not redistributed here.

To keep the repository small, only a sample of **8 images** is versioned in
`project/images/test_selected/` (`00000`, `00002`, `00005`, `00011`, `00012`, `00016`,
`00017`, `00018`; `00000` contains no sign panel and is kept as a negative example).
They keep their original names so they still match the annotations in `project/files/`.

- `project/files/salida_anotacion.csv` — ground truth for the 20 selected images, annotated
  with [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/).
- `project/files/correspondencia.txt` — mapping from the 20 selected images to the indices of
  the original course test set.

To run on the full set, copy the course test images into `project/images/test_selected/`
(the root-level `imagenesTest/` folder used in earlier versions is git-ignored).

The original assignment statement (PDF) was removed because it is course material and is
not ours to redistribute. Pipeline outputs (`a_gray_before` … `j_improve_images`,
`imagenesResultado/`) are no longer versioned; run the pipeline to regenerate them.
Older versions of the repository still contain the full image set in its history.

---

## Running

```bash
cd project/src
python main.py
```

Reads images from `project/images/test_selected/`, runs the full detection pipeline, and saves
all intermediate results to the `project/images/` subdirectories.

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
