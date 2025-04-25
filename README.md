# Fake News Detection Web App
 
Detect whether a news headline is real or fake in seconds, powered by a pre-trained ML model.

## Overview

This project is a Flask web application that:
- Cleans a news headline
- Vectorizes the text using TF-IDF
- Predicts if the headline is **Real** or **Fake**

## Training Data
The model is trained on [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset) data which had about 41,000+ titles and I have attached the source code of training model so you can see it if you want.

## Libraries 
Built with:
- Python
- Flask
- NLTK
- Scikit-learn

## ⚙️ How to Run

1. Install dependencies:
    ```bash
    pip install flask nltk scikit-learn joblib
    ```

2. Ensure `fake_news.pkl` and `tfid.pkl` are in your project directory.
3. Ensure that index.html is in the  `templates` folder which should be inside your working directory.

4. Run the Flask server:
    ```bash
    python app.py
    ```

5. Open your browser at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

