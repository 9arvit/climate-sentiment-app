from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    vector = vectorizer.transform([comment])
    prediction = model.predict(vector)[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return render_template('index.html', result=sentiment, comment=comment)

if __name__ == '__main__':
    app.run(debug=True)