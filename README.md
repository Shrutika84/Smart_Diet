# Smart Diet Recommender System

An AI-based personalized diet recommendation system built using machine learning and LLMs. It predicts the best diet plan for a patient based on health parameters and dynamically generates a full-day meal plan using prompt engineering with local LLMs via Ollama.

---

## 🚀 Features
- **Machine Learning Classifier** to predict diet types like `Low_Carb`, `Balanced`, `Low_Sodium`, etc.
- **Ollama-based Local LLM Integration** for generating dynamic meal plans.
- **Streamlit UI** for user-friendly input and instant recommendations.
- **End-to-End Pipeline**: from preprocessing to prediction and generation.

---

## 🧠 ML Model
- **Input Features**: Age, Gender, BMI, BP, Glucose, Activity, Disease Type, etc.
- **Model Used**: `RandomForestClassifier`
- **Training Data**: 1000 patient profiles with labeled diet recommendations.

---

## 🌐 How It Works
1. The user enters health details in the Streamlit UI.
2. The ML model predicts the most suitable diet type.
3. A local LLM (via Ollama) generates a full meal plan based on that diet type.
4. Output includes breakfast, lunch, dinner, snacks, and health tips.

---
## Result

![image](https://github.com/user-attachments/assets/37840b0a-afef-4b07-be8c-08346d83ee6d)


![image](https://github.com/user-attachments/assets/006fc00c-61e9-49e7-95dc-a33eb090d5e8)



## 📎 Dataset
Custom healthcare dataset with health indicators and target diet labels.  


