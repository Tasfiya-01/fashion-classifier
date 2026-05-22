# 👗 Fashion Classifier AI

A deep learning web app that classifies clothing images into 3 categories using **MobileNetV2 Transfer Learning**.

## 🎯 Results
| Metric | Value |
|--------|-------|
| Test Accuracy | **94.84%** |
| Val Accuracy | **94.84%** |
| Train Accuracy | **98.14%** |
| Classes | T-Shirt, Dress, Pants |
| Training Images | 1,440 |

## 🧠 Model Architecture
- **Base**: MobileNetV2 (pretrained on ImageNet)
- **Fine-tuned**: Last 30 layers unfrozen
- **Custom head**: GlobalAveragePooling → Dense(256) → Dropout → Dense(128) → Dropout → Softmax(3)
- **Dataset**: [Clothing Dataset](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full)

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
```
fashion-classifier/
├── app.py                # Streamlit web app
├── fashion_model.h5      # Trained model
├── classes.json          # Class label mapping
└── requirements.txt      # Dependencies
```

## 🌐 Live Demo
Deployed on Streamlit Cloud → [your-link-here]
