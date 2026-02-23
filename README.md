# 🛡️ SpamShield AI

AI-powered Spam Detection Web App built with Flask, scikit-learn, and OpenAI GPT.

## 🚀 Overview
SpamShield AI is a full-stack machine learning application that detects spam messages with **96%+ accuracy** using a Naive Bayes classifier and provides AI-generated explanations using OpenAI GPT.

## ✨ Features
- 🔐 User Authentication (Login/Register)
- 🤖 ML-based Spam Detection (Naive Bayes + TF-IDF)
- 🧠 GPT-powered intelligent explanations
- 📊 Interactive dashboard with history tracking
- 🎨 Responsive modern UI
- 🗂️ SQLite database integration

## 🛠️ Tech Stack
- **Backend:** Python, Flask, SQLAlchemy
- **ML:** scikit-learn, TF-IDF, Multinomial Naive Bayes
- **AI:** OpenAI API
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **Database:** SQLite

## ⚙️ Installation

```bash
git clone https://github.com/DishaAgarwalla/SpamShield_AI
cd SpamShield-AI
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

Create `.env` file:
```
OPENAI_API_KEY=your_key_here
SECRET_KEY=your_secret_key
```

Train model:
```bash
python train_model.py
```

Run app:
```bash
python app.py
```

Visit: http://127.0.0.1:5000/

## 📊 Model Performance
- Accuracy: **96.68%**
- Algorithm: Multinomial Naive Bayes

## 👩‍💻 Author
**Disha** – AI/ML Enthusiast 🚀

---
⭐ Star the repo if you found it helpful!
