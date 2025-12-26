# 🧠 Mental Health & Suicide Detection Using Social Media Posts

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.57%25-success)](/)

> An AI-powered system for early detection of mental health risks and suicidal intent using Natural Language Processing and Deep Learning

---

## 📌 Overview

In today's digital landscape, individuals increasingly express emotional distress, depression, and suicidal thoughts through social media platforms. However, manually identifying these critical warning signs at scale is virtually impossible.

This project presents an **end-to-end intelligent system** that automatically detects suicidal tendencies and mental health risks from textual content using advanced NLP, Machine Learning, and Deep Learning techniques.

**Key Achievement:** 96.57% accuracy with real-time prediction capabilities for immediate intervention support.

---

## 🎯 Objectives

- ✅ Detect suicidal and high-risk mental health language patterns from text  
- ✅ Perform deep linguistic and emotional analysis using NLP techniques  
- ✅ Train a high-performance deep learning model with robust evaluation  
- ✅ Provide a real-time prediction interface for practical deployment  
- ✅ Support early intervention and promote mental health awareness  

---

## 📂 Project Structure
```
mental-health-detection/
│
├── data/
│   ├── Suicide_Detection.csv
│   └── mental_health_cleaned.csv
│
├── models/
│   ├── best_mental_health_model.h5
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   ├── tfidf_vectorizer.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── Stage1_Data_Loading.ipynb
│   ├── Stage2_Preprocessing.ipynb
│   ├── Stage3_EDA.ipynb
│   ├── Stage4_Feature_Engineering.ipynb
│   ├── Stage5_Model_Training.ipynb
│   ├── Stage6_Evaluation.ipynb
│   └── Stage7_RealTime_Prediction.ipynb
│
├── app/
│   └── realtime_predictor.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🏗️ System Architecture
```
Raw Social Media Text
        ↓
Text Cleaning & Preprocessing (NLP)
        ↓
Exploratory Data Analysis & Sentiment Analysis
        ↓
Feature Engineering (TF-IDF + Emotional Features)
        ↓
Deep Learning Model (CNN + BiLSTM)
        ↓
Model Evaluation & Optimization
        ↓
Real-Time Prediction System
```

---

## 📊 Dataset

- **Source:** Kaggle – Suicide Watch Dataset
- **Records:** 232,000+ social media posts
- **Classes:**
  - `suicide` - Posts indicating suicidal intent or ideation
  - `non-suicide` - Posts with general content

The dataset is **balanced** to ensure unbiased model learning and fair evaluation across both classes.

---

## 🧼 NLP Preprocessing Pipeline

Our preprocessing pipeline includes:

1. **Text Normalization** - Lowercasing and contraction expansion
2. **Noise Removal** - Removing URLs, emails, hashtags, special characters
3. **Stopword Removal** - Filtering common words while preserving negations
4. **Tokenization & Lemmatization** - Breaking text into meaningful units
5. **Sentiment Extraction** - Using TextBlob for polarity analysis

---

## ⚙️ Feature Engineering

### TF-IDF Features
- Extracted **5,000 most important unigrams and bigrams**
- Captures semantic importance of words across the corpus

### Emotional & Behavioral Features
- Word count, character count, average word length
- Punctuation density (exclamation marks, question marks)
- Uppercase text ratio
- Mental health keyword frequency
- Sentiment polarity scores

**Final Feature Space:** 232,009 samples × 5,010 features

---

## 🧠 Deep Learning Model

### Hybrid CNN–BiLSTM Architecture

| Component | Purpose |
|-----------|---------|
| **CNN** | Captures local patterns and emotional phrases |
| **BiLSTM** | Understands long-term dependencies and context |
| **Dense Layers** | High-level feature abstraction and classification |

### Training Strategy

- **Cross-Validation:** 5-Fold stratified CV
- **Optimizer:** Adamax with adaptive learning rate
- **Regularization:** Early stopping, dropout layers
- **Class Balancing:** Weighted loss function

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.57% |
| **Precision** | 0.9650 |
| **Recall** | 0.9664 |
| **F1-Score** | 0.9657 |
| **ROC-AUC** | 0.99+ |

---

## 🚀 Real-Time Prediction System

The system provides instant mental health risk assessment:

```python
from realtime_predictor import predict_mental_health

# Example prediction
result = predict_mental_health(
    "I feel hopeless and don't want to live anymore",
    return_probabilities=True
)

print(result)
```

**Output:**
```
Prediction: suicide
Confidence: 98.9%
⚠️ HIGH RISK — Immediate intervention recommended
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- CUDA (optional, for GPU acceleration)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mental-health-detection.git
cd mental-health-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## 📖 Usage

### Training the Model

```bash
# Run notebooks in sequence
jupyter notebook notebooks/Stage1_Data_Loading.ipynb
# ... continue through Stage7
```

### Making Predictions

```python
from app.realtime_predictor import predict_mental_health

# Single prediction
text = "Your text here"
result = predict_mental_health(text)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 🌍 Real-World Applications

- **Suicide Prevention Platforms** - Early warning systems for crisis intervention
- **Social Media Monitoring** - Automated content moderation and user safety
- **Healthcare & Clinical Tools** - Screening and triage in mental health services
- **Crisis Helplines** - Prioritizing high-risk cases for immediate response
- **Research & Analytics** - Understanding mental health trends at scale

---

## 🔒 Ethical Considerations

This project is designed with the following ethical principles:

- **Privacy First** - No personal data collection or storage
- **Support, Not Replace** - Designed to assist, not replace, mental health professionals
- **Transparent AI** - Model decisions should be interpretable and explainable
- **No Discrimination** - Fair and unbiased across all demographics
- **Crisis Resources** - Always provide helpline information alongside predictions

**Important:** This system is a screening tool and should never be the sole basis for clinical decisions.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Crisis Resources

If you or someone you know is in crisis, please reach out:

- **National Suicide Prevention Lifeline (US):** 988 or 1-800-273-8255
- **Crisis Text Line:** Text HOME to 741741
- **International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Kaggle for providing the Suicide Watch Dataset
- Mental health professionals who validate the importance of this work
- Open-source community for excellent ML/NLP libraries

---

## 📚 References

1. Dataset: [Kaggle Suicide Watch Dataset](https://www.kaggle.com/)
2. World Health Organization - Mental Health Statistics
3. Research papers on NLP for mental health detection

---

<div align="center">

**⚠️ If you're struggling with mental health, please seek help from qualified professionals**

Made with ❤️ for mental health awareness

</div>
EOF

# Add to git
git add README.md

# Commit
git commit -m "Add comprehensive README documentation"

# Push to GitHub
git push origin main



