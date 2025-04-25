import joblib
from nltk.corpus import stopwords

model = joblib.load('fake_news.pkl')
vectorize = joblib.load('tfid.pkl')

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

title = ["Win Super Bonus with 1xbet"]

def predict_news(title):
    clean_title = [clean_text(title)]
    vectorize_title = vectorize.transform(clean_title).toarray()
    predict = model.predict(vectorize_title)
    return predict[0]

