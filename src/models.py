from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import numpy as np

from preprocess.preprocess_text import preprocess_text


def make_model():
    return make_random_forest()

def make_random_forest():
    return RandomForestClassifier(n_estimators=90, random_state=1)

def evaluate_model(X_raw, y):
    n_splits = 5

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []

    for train_index, test_index in kf.split(X_raw):
        X_train_raw, X_test_raw = X_raw.iloc[train_index], X_raw.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_processed = X_train_raw.apply(preprocess_text)
        vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(X_train_processed)

        X_test_processed = X_test_raw.apply(preprocess_text)
        X_test = vectorizer.transform(X_test_processed)

        model = make_model()
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        scores.append(score)

    print(f"{n_splits}-folds cross validation scores: {scores}")
    print(f"Mean accuracy: {np.mean(scores):.2f}")