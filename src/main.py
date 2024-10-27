import click
import joblib

from data import make_dataset
from feature import make_features
from models import make_model, evaluate_model
from preprocess.preprocess_text import preprocess_text


@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y, vectorizer = make_features(df)
    model = make_model()
    model.fit(X, y)

    joblib.dump((model, vectorizer), model_dump_filename)


@click.command()
@click.option("--input_filename", default="data/raw/test.csv", help="File testing data")
@click.option("--model_dump_filename", default="models/model_dump.pkl", help="File with the model doing the prediction")
@click.option("--output_filename", default="data/processed/predictions.csv", help="File with the output predictions")
def predict(input_filename, model_dump_filename, output_filename):
    model, vectorizer = joblib.load(model_dump_filename)
    df = make_dataset(input_filename)
    df['processed_text'] = df['video_name'].apply(preprocess_text)
    X = vectorizer.transform(df['processed_text'])

    predictions = model.predict(X)
    df['is_comic_prediction'] = predictions

    df.to_csv(output_filename, index=False)


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(input_filename):
    df = make_dataset(input_filename)
    X = df['video_name']
    y = df['is_comic']

    evaluate_model(X, y)


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
