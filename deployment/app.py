from flask import Flask, render_template, request
from model import predict_sentiment, recommend_products

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    sentiment = predict_sentiment(review)
    return render_template("index.html", prediction=sentiment)

@app.route('/recommend', methods=['POST'])
def recommend():
    username = request.form['username']
    recommendations = recommend_products(username)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
