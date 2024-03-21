from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

sia = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods = ['POST'])
def analyze():
    text = request.form['text']
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05 and sentiment_score['compound'] < 0.25:
        sentiment = 'Slightly Positive'
    elif sentiment_score['compound'] >= 0.25:
        sentiment = 'Positive'
    elif sentiment_score['compound'] <= -0.05 and sentiment_score['compound'] > -0.25:
        sentiment = 'Slightly Negative'
    elif sentiment_score['compound'] <= -0.25:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return render_template('result.html', text = text, sentiment = sentiment)

if __name__ == '__main__':
    app.run(debug = True)