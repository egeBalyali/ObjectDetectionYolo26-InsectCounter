# Grape Moth Detection on Sticky Traps — YOLO26-Large

A **single-class object detection** model for finding and counting grape moths (`grape_moth`) in sticky-trap images. The trained model is stored as `best.pt` (PyTorch/Ultralytics YOLO checkpoint).

**Try the deployed model:** [evren.ssyz.org.tr/inference](https://evren.ssyz.org.tr/inference)

> This model detects **grape moths only**. It does not identify other insects, provide instance segmentation masks, or estimate pest-treatment thresholds. The examples below assume `best.pt` can be loaded by a version of Ultralytics that supports **YOLO26**.

## Model overview

| Item | Value |
|---|---|
| Task | Object detection (bounding boxes) |
| Architecture | **YOLO26-Large** |
| Weights | `best.pt` |
| Input image size | **640 × 640** (`imgsz=640`, with preprocessing/letterboxing as needed) |
| Number of classes | **1** |
| Class | `0: grape_moth` |
| Dataset | [Insect Sticky Traps Rectified — version 1](https://universe.roboflow.com/vineyard/insect-sticky-traps-rectified/dataset/1) |
| Dataset license | **CC BY 4.0** (attribution required) |
| Dataset split | Train **92%**, validation **5%**, test **3%** |
| Deployment | https://evren.ssyz.org.tr/inference |

## Directory structure

Place `best.pt` alongside `README.md`. The dataset is kept in `images/`, with each split containing its own images and labels. For example:

```text
project/
├── best.pt
├── README.md
└── images/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

**Dataset YAML:** The Roboflow-exported configuration was:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['grape_moth']

roboflow:
  workspace: vineyard
  project: insect-sticky-traps-rectified
  version: 1
  license: CC BY 4.0
  url: https://universe.roboflow.com/vineyard/insect-sticky-traps-rectified/dataset/1
```

**Check the relative paths.** The `../train/images` style is typical of Roboflow exports. If your **actual** layout is exactly the tree above, where `data.yaml` and the split folders are all inside `images/`, change those three lines to:

```yaml
path: .
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['grape_moth']
```

When possible, run commands from the project root using `data=images/data.yaml`. Ensure paths resolve correctly in your environment; if necessary, replace `path: .` with the absolute path to the `images/` dataset directory. The train/val/test percentages are the **reported split configuration**; actual counts should be checked against the contents of each folder.

## Installation

Use a Python version supported by your Ultralytics/PyTorch installation, and install an **Ultralytics release that supports YOLO26**.

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install --upgrade ultralytics
```

For CUDA acceleration, install a PyTorch build appropriate for your GPU: [PyTorch setup](https://pytorch.org/get-started/locally/).

## Inference

### Command line

Run predictions on one image:

```bash
yolo detect predict model=best.pt source=trap.jpg imgsz=640 conf=0.25 save=True
```

Run predictions on a directory, video, or webcam:

```bash
yolo detect predict model=best.pt source=sample_traps/ imgsz=640 save=True
yolo detect predict model=best.pt source=trap_video.mp4 imgsz=640 save=True
yolo detect predict model=best.pt source=0 imgsz=640 show=True
```

Output is usually written under `runs/detect/predict*/`. The `conf=0.25` threshold is a **starting value**; tune it for the deployment camera and operating conditions.

### Python: detect and count grape moths

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="trap.jpg",
    imgsz=640,
    conf=0.25,
    save=True,
)

for result in results:
    moth_count = len(result.boxes) if result.boxes is not None else 0
    print(f"Grape moth detections: {moth_count}")

    if result.boxes is not None:
        for box in result.boxes:
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"grape_moth: {confidence:.3f}, box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

> Counts are detections **in one image**, not unique insects accumulated over time. The same trapped moth may appear in multiple photos.

## Training configuration

Settings reported by the training run:

| Parameter | Value |
|---|---:|
| Model | YOLO26-Large |
| Image resolution | 640 |
| Epochs configured/completed | 100 / 100 |
| Batch size | 32 |
| Early stopping patience | 50 epochs |
| Validation frequency | Every epoch |
| GPU count | 1 |
| Training/validation/test split | 92% / 5% / 3% |
| Best checkpoint epoch | **73** |
| Saved model | `best.pt` from epoch 73 |
| Dashboard training time | 4 min 14 sec |
| Displayed total time | ~5 minutes |
| GPU hours | 0.09 h |
| Platform cost | 0.9 CR |
| Peak GPU memory | 20.1 GB |
| GPU utilization | 82% |
| Displayed learning rate | 5.74 × 10⁻⁴ |
| Images processed (dashboard) | 13.5K |
| Reported throughput | 152 images/s |
| Average / fastest / slowest duration | 2.2 s / 2.0 s / 7.0 s |

The dashboard learning rate is a **displayed value**, not necessarily the initial learning rate. The 152 images/s figure is from the training dashboard, **not a verified production inference benchmark**. Although patience was set to 50, the run finished its configured 100 epochs.

### Example retraining command

If the project directory and YAML paths above are set up correctly:

```bash
yolo detect train model=best.pt data=images/data.yaml imgsz=640 epochs=100 batch=32 patience=50 val=True
```

This is an **illustrative fine-tuning command using the saved checkpoint**, not a complete reconstruction of the original training job. The original optimizer, seed, augmentations, and learning-rate schedule were not provided. In Ultralytics, validation during training is normally evaluated at the end of each epoch when enabled.

## Evaluation results

Values transcribed from the completed run's dashboard:

| Metric | Value |
|---|---:|
| **mAP@50** | **94.8%** |
| **mAP@50:95** | **48.5%** |
| Precision | 85.0% |
| Recall | 93.6% |
| F1 | 89.1% |
| Loss (dashboard) | 2.9865 |
| Best epoch | 73 |

- **mAP@50** measures average precision at an IoU threshold of 0.50; it does **not** mean 94.8% of all insects are found.
- **mAP@50:95** averages precision over IoU thresholds 0.50 through 0.95. Its lower value indicates tighter bounding-box localization is substantially harder.
- **Precision** reflects the fraction of predicted detections that are correct at the evaluation operating point.
- **Recall** reflects the fraction of annotated moths detected at that operating point.
- **F1** combines precision and recall.

These are the **reported evaluation metrics**; the screenshot does not establish whether they were calculated on the validation or held-out test subset. With a **3% test split**, assess the number of test images/instances and confidence intervals before claiming strong generalization.

To evaluate using the held-out test split:

```bash
yolo detect val model=best.pt data=images/data.yaml split=test imgsz=640
```

## Deployment

Online inference interface: **https://evren.ssyz.org.tr/inference**

Upload or provide an image through the site's interface to try the deployed model named yolo26BöcekSayacı. You can also run inference locally using the commands above.

For real-world sticky-trap monitoring, test on unseen camera setups, varying illumination, trap wear, and crowded/overlapping insects. Choosing a confidence threshold and creating alert rules are separate application decisions.

## Dataset attribution and license

**Dataset:** *Insect Sticky Traps Rectified*, version 1, published in the `vineyard` workspace on Roboflow Universe.  
**Source:** https://universe.roboflow.com/vineyard/insect-sticky-traps-rectified/dataset/1  
**Dataset license:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

The dataset's CC BY 4.0 license requires attribution and indicates obligations such as noting modifications when distributing derived materials. **Do not assume the trained model weights or this repository automatically have the same license**; specify a separate license for your code and model weights if you publish them.

## Limitations

- Supports **only `grape_moth`**, not all pests seen on traps.
- Performs **bounding-box object detection**, not semantic/instance segmentation or oriented bounding boxes.
- Small, crowded, partially occluded, or visually unusual moths can be missed or double-counted.
- Results may degrade on images unlike the training distribution.
- The model does not independently recommend pesticide application or infer economic damage thresholds.
