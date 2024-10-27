import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer


def preprocess_text(text):
    nltk.download('stopwords', quiet=True)
    french_stop_words = stopwords.words('french')

    stemmer = FrenchStemmer()

    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in french_stop_words]
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)