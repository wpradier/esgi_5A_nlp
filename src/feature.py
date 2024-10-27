from sklearn.feature_extraction.text import CountVectorizer
from preprocess.preprocess_text import preprocess_text


def make_features(df):
    y = df["is_comic"] if "is_comic" in df.columns else None
    df['processed_text'] = df['video_name'].apply(preprocess_text)

    vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['processed_text'])

    return X, y, vectorizer
