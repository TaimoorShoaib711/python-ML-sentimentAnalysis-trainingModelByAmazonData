import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import string
from tqdm import tqdm
import joblib
import os
from sklearn.utils import resample


MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "text_vectorizer.pkl"
INFO_PATH = "model_info.txt"


# Clean and simplify text
def preprocess_text(text_series):
    return text_series.apply(
        lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else ""
    )


# Convert 1–5 scores → 0, 1, 2 (Negative / Neutral / Positive)
def simplify_score(score):
    if score <= 2:
        return 0
    elif score == 3:
        return 1
    else:
        return 2
def balance_dataset(df):
    df['Sentiment'] = df['Score'].apply(simplify_score)

    # Separate classes
    df_negative = df[df['Sentiment'] == 0]
    df_neutral  = df[df['Sentiment'] == 1]
    df_positive = df[df['Sentiment'] == 2]

    # Upsample Negative and Neutral to match Positive
    df_negative_upsampled = resample(df_negative, replace=True, n_samples=len(df_positive), random_state=42)
    df_neutral_upsampled  = resample(df_neutral, replace=True, n_samples=len(df_positive), random_state=42)

    # Combine all and overwrite df
    df = pd.concat([df_positive, df_negative_upsampled, df_neutral_upsampled]).reset_index(drop=True)
    return df

# Train model and save it
def train_and_save_model(df):
    print("🧹 Preprocessing text...")
    df['Text'] = preprocess_text(df['Text'])
    df = balance_dataset(df)

    df['Sentiment'] = df['Score'].apply(simplify_score)

    X = df['Text']
    y = df['Sentiment']

    print(" Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(" Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("\n Training Logistic Regression model...\n")
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Model Accuracy: {acc * 100:.2f}%")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]))

    # Save model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # Save info (rows)
    with open(INFO_PATH, "w") as f:
        f.write(str(len(df)))

    print("\n Model and vectorizer saved successfully!")
    return model, vectorizer


#  Load model or retrain if data changed
def load_or_train(df):
    retrain = False
    current_rows = len(df)

    if all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, INFO_PATH]):
        with open(INFO_PATH, "r") as f:
            saved_rows = int(f.read().strip() or 0)

        if saved_rows != current_rows:
            print(f"Data size changed ({saved_rows} → {current_rows}). Retraining model...")
            retrain = True
        else:
            print(" Found saved model! Loading from disk...")
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        retrain = True

    if retrain:
        print(" Training new model...")
        model, vectorizer = train_and_save_model(df)

    return model, vectorizer


# Convert label number → emoji sentiment
def get_sentiment_label(label):
    if label == 0:
        return "Negative 😠"
    elif label == 1:
        return "Neutral 😐"
    else:
        return "Positive 😊"


# Main
def main():
    df = pd.read_csv("amazonReviews/Reviews.csv", nrows=10000)
    print(f" Loaded {len(df)} rows.")

    model, vectorizer = load_or_train(df)

    print("\n Making example predictions...\n")
    samples = [
        "This product was absolutely amazing, I loved it!",
        "Terrible quality, waste of money.",
        "It was okay, nothing special.",
        "Fantastic flavor and packaging.",
        "Broke after one use, not worth it."
    ]

    X_samples = vectorizer.transform(samples)
    preds = model.predict(X_samples)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_samples)
        confidences = probs.max(axis=1)
    else:
        confidences = np.ones(len(samples))

    for text, pred, conf in zip(samples, preds, confidences):
        sentiment = get_sentiment_label(pred)
        print(f"Review: {text}\n Sentiment: {sentiment}\n Confidence: {conf * 100:.2f}%\n")


if __name__ == "__main__":
    main()
