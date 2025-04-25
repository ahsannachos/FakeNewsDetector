from flask import Flask, request, render_template
from predictor import predict_news

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', method=['POST'])
def predict():
    if request.method == 'POST':
        newstitles=request.form('newstitles')
        predictions = predict_news(newstitles)

        if predictions == 1
            result = "Realllleee"
        else:
            result = "Fakeeee"

        return render_template('index.html', result = result)

if __name__ == '__main__':
    app.run(debug=True)            


