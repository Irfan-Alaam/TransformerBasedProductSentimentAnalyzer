import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import IsolationForest
import nltk

nltk.download('vader_lexicon')

def load_dataset(filepath="amazon_reviews_20250606_073201.csv"):
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Combine review_title and body into a single text column
    df['text'] = df['review_title'] + ' ' + df['body']
    
    # Ensure rating is numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna(subset=['text', 'rating'])
    
    return df[['text', 'rating', 'date']]

def analyze_sentiments(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    df['sentiment_label'] = df['sentiment'].apply(
        lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral')
    return df

def detect_fake_reviews(df):
    # Feature engineering
    df['word_count'] = df['text'].str.split().str.len()
    df['rating_sentiment_diff'] = abs(df['rating'] / 5 - (df['sentiment'] + 1) / 2)

    # Anomaly detection
    if len(df) > 10:
        clf = IsolationForest(contamination=0.1, random_state=42)
        features = df[['rating', 'sentiment', 'word_count', 'rating_sentiment_diff']]
        df['is_fake'] = clf.fit_predict(features)
        df['is_fake'] = df['is_fake'].map({1: False, -1: True})
    else:
        df['is_fake'] = False
    return df