# AI-Future-Directions

## Overview
This repository contains the deliverables for **Task 1 – Edge AI**, which explores building a lightweight waste-classification model that can run on constrained hardware. The workflow is implemented in the notebook `Task1-Edge-AI/Edge_AI_Prototype.ipynb` and produces a quantized TensorFlow Lite artifact (`model.tflite`) suitable for deployment on embedded devices.

## Repository layout
- `Task1-Edge-AI/Edge_AI_Prototype.ipynb` – end-to-end pipeline (training → TFLite conversion → smoke tests → metrics).
- `Task1-Edge-AI/Image_Dataset/TrashType_Image_Dataset/` – six-class trash dataset (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`). Images are organized per class for use with `ImageDataGenerator`.
- `Task1-Edge-AI/model.tflite` – MobileNetV2 backbone converted and size-optimized via post-training quantization.
- `Answers.pdf` – written responses for the broader AI Future Directions assignment (see PDF for context).

## Project folder structure
```
AI-Future-Directions/
├── Answers.pdf
└── Task1-Edge-AI/
    ├── Edge_AI_Prototype.ipynb
    ├── Image_Dataset/
    │   └── TrashType_Image_Dataset/
    │       ├── cardboard/
    │       ├── glass/
    │       ├── metal/
    │       ├── paper/
    │       ├── plastic/
    │       └── trash/
    └── model.tflite
```

## Prerequisites
1. Python 3.10+ with the following key packages: `tensorflow>=2.13`, `tensorflow-datasets`, `pillow`, `numpy`, `scikit-learn`.
2. (Recommended) GPU-enabled environment or Google Colab for shorter training time.
3. Adequate disk space (~3 GB) for the dataset and intermediate checkpoints.

Install dependencies locally:
```bash
python -m venv .venv
.venv\Scripts\activate  # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt  # or pip install tensorflow pillow numpy scikit-learn
```

> **Dataset note:** The TrashType dataset can be sourced from Kaggle (`https://www.kaggle.com/datasets/garythung/trashnet`) or other public mirrors. Once downloaded, keep the folder name `TrashType_Image_Dataset` as expected by the notebook.

## Running the notebook
1. Launch Jupyter or open the notebook directly in Colab.
2. Update the `data_dir` path in the “Setup” cell if you place the dataset elsewhere.
3. Run cells sequentially:
   - **Part 1:** Loads the dataset with `ImageDataGenerator`, fine-tunes `MobileNetV2`, and trains for 10 epochs at 128×128 resolution.
   - **Part 2:** Converts the trained Keras model to TFLite using default optimization (`tf.lite.Optimize.DEFAULT`).
   - **Part 3:** Performs a brute-force sanity test by iterating over every image and recording predictions.
   - **Part 4:** Computes validation metrics (`classification_report`, `confusion_matrix`).
4. Download the produced `model.tflite` (Colab: `Files` sidebar → right-click → *Download*).

## Current results
- Training accuracy climbs to ~0.93, but validation accuracy remains low (~0.14) because the naive fine-tuning overfits to the training split.
- The confusion matrix shows the model is biased toward the `metal` class. This is visible in the brute-force test logs where most cardboard images are misclassified as metal.

```startLine:endLine:Task1-Edge-AI/Edge_AI_Prototype.ipynb
cardboard_148.jpg | True: cardboard → Predicted: metal (0.99)
# ...
cardboard_034.jpg | True: cardboard → Predicted: metal (1.0)
```

## Suggested improvements
- **Data balancing & augmentation:** Apply class-specific augmentations or resampling to mitigate class imbalance.
- **Longer fine-tuning:** Unfreeze upper MobileNetV2 blocks and use a lower learning rate schedule (e.g., cosine decay) with early stopping.
- **Validation strategy:** Adopt stratified K-fold cross-validation for a more reliable estimate than a single split.
- **Edge deployment checks:** Profile the exported `.tflite` model on representative hardware (e.g., Coral Dev Board, Raspberry Pi) using `tflite_runtime` or LiteRT.
- **Automated evaluation:** Capture aggregate metrics (accuracy, F1, latency) during TFLite inference instead of console-only output.

## Reproducing TFLite testing standalone
```bash
python - <<'PY'
import tensorflow as tf, numpy as np, os
from PIL import Image

interpreter = tf.lite.Interpreter(model_path="Task1-Edge-AI/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = ['cardboard','glass','metal','paper','plastic','trash']
img = Image.open("Task1-Edge-AI/Image_Dataset/TrashType_Image_Dataset/cardboard/cardboard_001.jpg").convert('RGB').resize((128,128))
x = np.expand_dims(np.array(img)/255.0, 0).astype('float32')
interpreter.set_tensor(input_details[0]['index'], x)
interpreter.invoke()
pred = interpreter.get_tensor(output_details[0]['index'])[0]
print(dict(zip(class_names, pred.round(3))))
PY
```

## Version control hygiene
- The raw dataset directory (`Task1-Edge-AI/Image_Dataset/`) is now excluded via `.gitignore` to keep future commits lightweight. Recreate the folder locally before running the notebook.
- Large binary artifacts (`model.tflite`, dataset images) are configured for Git LFS in `.gitattributes`. Run `git lfs install` once per machine and commit via LFS if you truly need to version refreshed assets.

## Next steps
- Track experiments with notebook metadata or Weights & Biases.
- Script the pipeline (e.g., `train.py`, `export.py`) for CI automation.
- Integrate LiteRT for forward compatibility once TensorFlow 2.20 drops `tf.lite.Interpreter`.

