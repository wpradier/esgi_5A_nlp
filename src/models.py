from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


def make_model():
    return Pipeline(
        [
            ("count_vectorizer", CountVectorizer(min_df=5)),
            ("random_forest", RandomForestClassifier(n_estimators=100, min_samples_split=10)),
        ]
    )

    # return CustomModel(
    #     classifier=RandomForestClassifier(n_estimators=100),
    #     vectorizer=CountVectorizer(min_df=5),
    # )


class CustomModel:
    def __init__(self, classifier, vectorizer):
        self.classifier = classifier
        self.vectorizer = vectorizer


    def fit(self, texts, Y):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, y)
        return self

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)
