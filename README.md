# Leukemia White Blood Cell Image Classification

This project implements a lightweight GUI application using EfficientNetB3 with Extra Trees and Naive Bayes classifiers for classifying leukemia white blood cell images. It uses transfer learning via a VGG16 backbone for feature extraction.

## Features

- GUI built with `Tkinter`
- Image feature extraction using `VGG16` from Keras
- Classification using:
  - Gaussian Naive Bayes
  - Extra Trees Classifier (EfficientNetB3)
- Real-time image prediction
- Confusion matrix and performance metric visualization

## Requirements

Install required packages using the `requeriment.txt` file:

### Note:

* TensorFlow 1.14.0 and Keras 2.3.1 are used (consider using a virtual environment).
* The project relies on `VGG16` for feature extraction (not actually EfficientNetB3, though named as such).

## Files

* `main.py`: Main script to run the GUI and models
* `naive_bayes_model.joblib`: Pretrained Gaussian Naive Bayes model
* `EfficientNetB3_extra_trees_model.joblib`: Pretrained Extra Trees classifier model
* `requeriment.txt`: Required Python packages

## Dataset

Expected folder structure under `dataset/`:

```
dataset/
├── ALL/
│   ├── image1.jpg
│   └── ...
└── normal/
    ├── image2.jpg
    └── ...
```

Each folder should contain images of their respective classes.

## How to Use

1. Run the GUI:

```bash
python main.py
```

2. Buttons & Actions:

   * **Upload Dataset**: Select the dataset directory.
   * **Image Preprocessing**: Extract features and save them as `.npy` files.
   * **Build & Train NBC**: Train or load the Naive Bayes Classifier.
   * **Build & Train EfficientNetB3**: Train or load the Extra Trees Classifier.
   * **Upload Test Image**: Select an image for prediction.
   * **Performance Evaluation**: Display comparison graphs.
   * **Exit**: Close the application.

## Output

The GUI provides:

* Textual feedback in a scrolling textbox
* Real-time predictions on images
* Visual performance metrics and confusion matrices


