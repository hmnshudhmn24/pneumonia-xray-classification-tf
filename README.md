# Pneumonia Detection from Chest X-Rays

This project uses TensorFlow and transfer learning (EfficientNetB0) to classify chest X-ray images as Pneumonia or Normal.

## 📁 Dataset
You can download the dataset from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## 🧠 Model
- Pretrained EfficientNetB0 from TensorFlow Keras Applications
- Binary classification (Normal vs Pneumonia)
- Data augmentation and early stopping

## 🚀 How to Run

1. Install dependencies:
```bash
pip install tensorflow matplotlib numpy
```

2. Place the dataset in a `chest_xray` directory.

3. Run the script:
```bash
python pneumonia_detection.py
```

## 📊 Evaluation
- Accuracy, Precision, Recall
- Confusion Matrix

## 📌 Note
Ensure the dataset is in the following structure:

```
chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

## 🧑‍💻 Author
Developed using TensorFlow 2.x and EfficientNet.