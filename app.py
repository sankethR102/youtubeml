from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and process dataset once (global)
df = pd.read_csv("INvideos.csv")
df["description"] = df["description"].fillna("")

# Train a simple TF-IDF + Linear Regression model
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df["description"])
y = df["views"]

model = LinearRegression()
model.fit(X, y)

@app.route('/')
def index():
    # Extra columns
    df["contains_capitalized"] = df["title"].apply(lambda x: any(w.isupper() for w in x.split()))
    df["title_length"] = df["title"].apply(lambda x: len(x))

    # Plot 1
    plt.figure(figsize=(10, 5))
    sns.histplot(df["title_length"], kde=False, bins=30, color="#003f5c")
    plt.title("Distribution of Title Lengths")
    plt.xlabel("Title Length")
    plt.ylabel("Number of Videos")
    plt.tight_layout()
    plt.savefig('static/plots/title_length_distribution.png')
    plt.close()

    # Plot 2
    plt.figure(figsize=(10, 5))
    plt.scatter(df["views"], df["title_length"], alpha=0.5, color="#FF5722")
    plt.title("Views vs. Title Length")
    plt.xlabel("Views")
    plt.ylabel("Title Length")
    plt.tight_layout()
    plt.savefig('static/plots/views_vs_title_length.png')
    plt.close()

    # Plot 3
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='cubehelix')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig('static/plots/correlation_matrix.png')
    plt.close()

    # Plot 4
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
