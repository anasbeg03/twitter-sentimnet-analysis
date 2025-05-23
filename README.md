# 💬 Twitter Sentiment Analysis 🧠

A machine learning and deep learning-based NLP project to classify the sentiment of tweets as **positive** or **negative** using cleaned text data.

---

## 📌 Project Objective

To develop a pipeline that can classify the sentiment of tweets using various machine learning and deep learning models. This includes cleaning the data, performing exploratory analysis, vectorizing text, training models, and evaluating their performance.

---

## 🗂️ Dataset Summary

- **Source:** Twitter sentiment CSV dataset
- **Columns:**
  - `text`: Raw tweet
  - `sentiment`: Target label (`positive`, `negative`)
- **Size:** ~100,000 tweets

---

## ⚙️ Workflow Overview

1. **Data Cleaning**
   - Lowercasing, removing URLs, mentions, hashtags, emojis, punctuation
2. **Exploratory Data Analysis (EDA)**
   - Sentiment distribution, tweet length histograms, frequent words
3. **Text Preprocessing**
   - Tokenization, stopword removal
4. **Feature Engineering**
   - TF-IDF Vectorizer (for ML)
   - Tokenization + Padding (for DL)
5. **Modeling**
   - Logistic Regression
   - Random Forest
   - LSTM Neural Network
6. **Evaluation**
   - Accuracy, Precision, Recall, F1 Score, ROC-AUC
   - Confusion Matrix & ROC curves
7. **Visualization**
   - WordClouds, confusion matrix heatmaps, performance bars

---

## 🚀 Models and Performance

| Model         | Vectorizer         | Accuracy |
|---------------|--------------------|----------|
| Logistic Regression | TF-IDF       | ~82%     |
| Random Forest        | TF-IDF       | ~84%     |
| LSTM Neural Network  | Tokenized + Padded | ~87%     |

---

## 📊 Visualizations

- WordClouds for positive/negative tweets
- Tweet length distribution
- ROC Curves
- Confusion Matrix
- Bar plots of metrics

---

## 🛠 Tech Stack

- **Language:** Python
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `keras`, `tensorflow`, `nltk`, `wordcloud`

---

## 📁 Project Structure

