# --- Add this BEFORE importing pyplot ---
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for Flask apps

import matplotlib.pyplot as plt
# ----------------------------------------

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

DATA_FILE = "INvideos.csv"
MODEL_FILE = "view_predictor.joblib"
VECTORIZER_FILE = "tfidf_vectorizer.joblib"

# Load dataset and preprocess
df = pd.read_csv(DATA_FILE)
df["description"] = df["description"].fillna("")

# Function to train and save model/vectorizer
def train_and_save():
    print("Training model... This may take a moment.")
    sample_size = 10000  # limit sample size for quick training
    df_sample = df.sample(sample_size, random_state=42)

    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df_sample["description"])
    y = df_sample["views"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Save model and vectorizer
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print("Model and vectorizer saved.")
    return model, vectorizer

# Load or train model and vectorizer
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    print("Loaded saved model and vectorizer.")
else:
    model, vectorizer = train_and_save()

@app.route('/')
def index():
    # Extra columns for plots
    df["contains_capitalized"] = df["title"].apply(lambda x: any(w.isupper() for w in x.split()))
    df["title_length"] = df["title"].apply(len)

    # Plot 1: Title Length Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df["title_length"], kde=False, bins=30, color="#003f5c")
    plt.title("Distribution of Title Lengths")
    plt.xlabel("Title Length")
    plt.ylabel("Number of Videos")
    plt.tight_layout()
    plt.savefig('static/plots/title_length_distribution.png')
    plt.close()

    # Plot 2: Views vs Title Length
    plt.figure(figsize=(10, 5))
    plt.scatter(df["view_count"], df["title_length"], alpha=0.5, color="#FF5722")
    plt.title("Views vs. Title Length")
    plt.xlabel("Views")
    plt.ylabel("Title Length")
    plt.tight_layout()
    plt.savefig('static/plots/views_vs_title_length.png')
    plt.close()

    # Plot 3: Correlation Matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='cubehelix')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig('static/plots/correlation_matrix.png')
    plt.close()

    # Plot 4: Word Cloud of Titles
    all_titles = " ".join(df["title"])
    wc = WordCloud(width=1200, height=500, background_color="white", collocations=False).generate(all_titles)
    plt.figure(figsize=(15, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Common Words in Video Titles")
    plt.tight_layout()
    plt.savefig('static/plots/word_cloud.png')
    plt.close()

    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        desc = request.form['description']
        if desc.strip():
            vec = vectorizer.transform([desc])
            pred_views = model.predict(vec)[0]
            prediction = f"{int(pred_views):,} views (estimated)"
    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
