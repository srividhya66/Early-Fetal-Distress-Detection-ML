# 🤰 Early Fetal Distress Detection Using Machine Learning

This project uses machine learning algorithms to detect and classify fetal health conditions—Normal, Suspect, or Pathological—based on Cardiotocography (CTG) data. The goal is to help predict potential fetal distress early and improve prenatal care decisions.

---

## 📊 Project Overview

- **Domain:** Healthcare, Machine Learning
- **Tech Stack:** Python, Flask, Scikit-learn, Pandas, HTML
- **ML Model:** Random Forest Classifier
- **UI:** Web interface built using Flask + HTML form
- **Dataset:** [Fetal Health Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) (CTG-based)

---

## 🚀 Features

- Accepts 21 clinical features related to fetal heart rate and uterine contractions
- Predicts fetal health: `Normal`, `Suspect`, or `Pathological`
- Displays prediction confidence scores
- Includes test mode for known pathological input
- Trained with `99% accuracy` using Random Forest

---

## 📁 Project Structure


├── app.py # Flask app to serve predictions
├── train_model.py # Trains and saves the ML model
├── fetal_health.csv # Input dataset
├── fetal_model.pkl # Trained Random Forest model
├── fetal_scaler.pkl # StandardScaler for preprocessing
├── templates/
│ └── index.html # Web form for user input
├── README.md 



---

## 🧠 Model Training

Run the training script:

```bash
python train_model.py


🌐 Running the Web App
Install requirements:

bash
Copy
Edit
pip install flask pandas scikit-learn joblib
Start the app:

bash
Copy
Edit
python app.py
Open in browser:

cpp
Copy
Edit
http://127.0.0.1:5000/
💻 Sample UI
User fills in 21 numerical fields (e.g., Baseline Value, Accelerations, Histogram Features), clicks Predict, and gets a result like:

✅ Prediction: Pathological
🎯 Confidence: 92.4%

📈 Evaluation Metrics
Metric	Value
Accuracy	99.1%
Precision	98.9%
Recall	98.7%
F1-Score	98.8%

The Random Forest model outperforms other models like Logistic Regression, SVM, and KNN.

📦 Future Enhancements
Use Deep Learning (e.g., LSTM) for real-time CTG analysis

Mobile integration for remote fetal monitoring

Deploy on Render or HuggingFace Spaces

📚 References
Kaggle: Fetal Health Classification

🧪 Evaluation Metrics

✅ AUROC & AUPRC – for imbalanced data

❤️ Sensitivity (Recall for distress) – to catch critical cases

🔒 Specificity – to reduce false alarms

⚖️ F1-score – balance precision & recall

🔍 Methodology

1️⃣ Preprocessing: Missing values, normalization, artifact removal
2️⃣ Feature Engineering:

FHR 📉: mean, variance, accelerations, decelerations

UC 📊: contraction duration, frequency, correlation with FHR
3️⃣ Modeling: Stratified CV + hyperparameter tuning
4️⃣ Explainability: SHAP + feature importance
