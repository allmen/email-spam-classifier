# 📧 Spam Email Classification

A powerful spam email classifier combining **traditional machine learning** and **deep learning (BERT, DistilBERT)** to detect spam with high accuracy. Includes a full pipeline from preprocessing to deployment with an interactive web interface.

## 🚀 Quick Start

### 🔧 Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/allmen/email-spam-classifier.git
cd spam-classifier
pip install -r requirements.txt
```

Download NLTK data:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### ⚙️ Train Models

```bash
# Train all models (ML + DL)
python train_models.py

# Train only ML (faster)
python train_models.py --no-dl
```

### 📊 Evaluate Models

```bash
python evaluate_models.py
```

### 🌐 Run Web App

```bash
python app.py
# Visit http://localhost:5000
```

## 📁 Key Files

- `train_models.py`: Train spam classifiers
- `evaluate_models.py`: Evaluate performance
- `app.py`: Flask web app
- `notebooks/`: Interactive notebook
- `web/templates/index.html`: UI page

## 🧠 Models Used

- **Traditional ML**: Naive Bayes, SVM, Random Forest, Logistic Regression
- **Deep Learning**: BERT, DistilBERT
- **Ensemble**: Combines ML and DL for best results

---

## 💡 Sample Prediction (Python API)

```python
from src.traditional_ml import TraditionalMLModels
from src.deep_learning import DeepLearningModels, EnsemblePredictor

ensemble = EnsemblePredictor(...)
result = ensemble.predict("URGENT! Your account is compromised.")
print(result)
```

---

## 🤝 Contributors

Special thanks to the contributors of this project:

- **[@usmaino](https://github.com/usmaino)** – ML integration and research
- **[@allmen](https://github.com/allmen)** – Deep learning model tuning
- **[@bitsmart](https://github.com/bitsmart)** – Web deployment and documentation

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co) for their pre-trained transformer models
- [SpamAssassin](https://spamassassin.apache.org) for foundational spam detection datasets and techniques

**Built with ❤️ by a passionate team for email security and AI learning.**
