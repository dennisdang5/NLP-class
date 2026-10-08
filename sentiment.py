"""CSCI 544 HW1, Fall 2026. Complete and submit this file as sentiment.py.

Implement the documented functions; you may add helpers and imports.
Use the handout and requirements.txt. Do not download resources at runtime.
This is an unimplemented student starter, not a solution or an autograder.
"""

import pandas as pd
import csv
import re
import html
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Lazy loading of stopwords for preprocessing section
_STOP_WORDS = None
_LEMMATIZER = None

def _average_len(texts):
    total = 0
    for t in texts:
        total += len(t)

    return total / len(texts)

def prepare_data(path, n_per_class=100000):
    """Return (train_df, test_df, stats), as specified in Section 1/interface."""
    raw_data = pd.read_csv(path, sep='\t', dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, on_bad_lines='skip', usecols=['review_body', 'star_rating'])
    df = raw_data.rename(columns={'review_body': 'review', 'star_rating': 'rating'})[['review', 'rating']]

    ratings = pd.to_numeric(df['rating'], errors='coerce')
    valid_rows = ratings.isin([1, 2, 3, 4, 5]) & (df['review'].str.strip() != '')
    excluded_rows = int((~valid_rows).sum())

    data = df.loc[valid_rows, ['review']].copy()
    data['rating'] = ratings[valid_rows].astype(int)

    rating_counts = {}

    for r in range(1, 6):
        is_r = data['rating'] == r
        count = is_r.sum()
        rating_counts[r] = int(count)

    class_counts = {
        "positive": rating_counts[4] + rating_counts[5],
        "negative": rating_counts[1] + rating_counts[2],
        "neutral": rating_counts[3],
    }

    data = data[data["rating"] != 3].copy()
    data["label"] = (data["rating"] >= 4).astype(int)

    positive_data = data[data["label"] == 1].sample(n=n_per_class, replace=False, random_state=42)
    negative_data = data[data["label"] == 0].sample(n=n_per_class, replace=False, random_state=42)

    sampled_data = pd.concat([positive_data, negative_data]).reset_index(drop=True)
    train_df, test_df = train_test_split(sampled_data, test_size=0.2, stratify=sampled_data['label'], random_state=42)

    stats = {'rating_counts': rating_counts,
             'class_counts': class_counts,
             'excluded_rows': excluded_rows}

    return train_df, test_df, stats

def clean_review(text):
    """Return one cleaned string, including the empty string when applicable."""
    text = html.unescape(text).lower()
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = contractions.fix(text)
    text = re.sub(r'[^a-z\s]', ' ', text)

    return ' '.join(text.split())


def preprocess_review(text):
    """Return stopword-filtered, noun-lemmatized text from cleaned input."""
    global _STOP_WORDS, _LEMMATIZER
    if _STOP_WORDS is None:
        _STOP_WORDS = set(stopwords.words('english')) - {'no', 'nor', 'not'}
        _LEMMATIZER = WordNetLemmatizer()

    tokens = []
    for word in text.split():
        if word not in _STOP_WORDS:
            tokens.append(word)

    lemmas = []
    for word in tokens:
        lemmas.append(_LEMMATIZER.lemmatize(word))

    return ' '.join(lemmas)


def extract_features(train_texts, test_texts):
    """Return (fitted_vectorizer, sparse_X_train, sparse_X_test)."""
    vectorizer = TfidfVectorizer()
    sparse_X_train = vectorizer.fit_transform(train_texts)
    sparse_X_test = vectorizer.transform(test_texts)
    return vectorizer, sparse_X_train, sparse_X_test


def train_model(name, X_train, y_train):
    """Return a fitted classifier for one of the three specified model names."""
    if name == 'Perceptron':
        model = Perceptron(random_state=42, max_iter=1000, tol=1e-3)
    elif name == 'Logistic Regression':
        model = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000, tol=1e-4)
    elif name == 'Naive Bayes':
        model = MultinomialNB(alpha=1.0)

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y):
    """Return unrounded accuracy, precision, recall, f1 in a dictionary."""
    y_prediction = model.predict(X)
    accuracy = float(accuracy_score(y, y_prediction))
    precision = float(precision_score(y, y_prediction, average='binary', pos_label=1, zero_division=0))
    recall = float(recall_score(y, y_prediction, average='binary', pos_label=1, zero_division=0))
    f1 = float(f1_score(y, y_prediction, average='binary', pos_label=1, zero_division=0))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def fit_bigram(train_texts, min_count=3):
    """Return a dictionary with vocab, bigrams, contexts; omit zero counts."""
    word_counts = Counter()
    for text in train_texts:
        for word in text.split():
            word_counts[word] += 1

    kept_words = set() # keep words seen at least min_count
    for word, count in word_counts.items():
        if count >= min_count:
            kept_words.add(word)
    vocab = kept_words | {'<unk>', '</s>'}

    bigrams = Counter()
    contexts = Counter()

    for text in train_texts:
        sequence = ['<s>']
        for word in text.split():
            if word in kept_words:
                sequence.append(word)
            else:
                sequence.append('<unk>')
        sequence.append('</s>')

        for i in range(len(sequence) - 1):
            context, word = sequence[i], sequence[i + 1]
            bigrams[(context, word)] += 1
            contexts[context] += 1

    return {
        'vocab': vocab,
        'bigrams': bigrams,
        'contexts': contexts
    }


def evaluate_bigram(model, test_texts, k=1.0):
    """Return a dictionary with unseen, n_tokens, add-k perplexity; do not refit."""
    vocab = model['vocab']
    bigrams = model['bigrams']
    contexts = model['contexts']
    vocab_len = len(vocab)
    kept = vocab - {'<unk>', '</s>'}

    unseen = 0
    n_tokens = 0
    log_probability_sum = 0.0

    for text in test_texts:
        sequence = ['<s>']
        for word in text.split():
            if word in kept:
                sequence.append(word)
            else:
                sequence.append('<unk>')
        sequence.append('</s>')

        for i in range(len(sequence) - 1):
            context, word = sequence[i], sequence[i + 1]
            c_hw = bigrams.get((context, word), 0)
            c_h = contexts.get(context, 0)
            if c_hw == 0:
                unseen += 1
            probability = (c_hw + k) / (c_h + k * vocab_len)
            log_probability_sum += math.log(probability)
            n_tokens += 1

    perplexity = math.exp(-log_probability_sum / n_tokens)
    return {
        'unseen': unseen,
        'n_tokens': n_tokens,
        'perplexity': perplexity
    }


def run_experiment(path="data.tsv", n_per_class=100000,
                   lm_train_size=5000, lm_test_size=1000):
    """Run the pipeline, print the 37 required lines, and return None."""
    # Section 1
    train_df, test_df, stats = prepare_data(path, n_per_class)
    counts = stats['class_counts']
    print(f'Positive reviews: {counts["positive"]}')
    print(f'Negative reviews: {counts["negative"]}')
    print(f'Neutral reviews: {counts["neutral"]}')

    # Section 2 and 3
    train_raw = train_df['review'].tolist()
    test_raw = test_df['review'].tolist()

    train_clean = []
    for text in train_raw:
        train_clean.append(clean_review(text))

    test_clean = []
    for text in test_raw:
        test_clean.append(clean_review(text))

    train_prep = []
    for text in train_clean:
        train_prep.append(preprocess_review(text))

    test_prep = []
    for text in test_clean:
        test_prep.append(preprocess_review(text))

    print(f'Average length before cleaning: {_average_len(train_raw):.4f}')
    print(f'Average length after cleaning: {_average_len(train_clean):.4f}')
    print(f'Average length before preprocessing: {_average_len(train_clean):.4f}')
    print(f'Average length after preprocessing: {_average_len(train_prep):.4f}')

    # Section 4 to 7
    _, X_train, X_test = extract_features(train_prep, test_prep)
    print(f'TF-IDF matrix shape: {X_train.shape}')

    y_train = train_df['label'].to_numpy()
    y_test = test_df['label'].to_numpy()

    for name in ['Perceptron', 'Logistic Regression', 'Naive Bayes']:
        model = train_model(name, X_train, y_train)

        train_metrics = evaluate_model(model, X_train, y_train)
        print(f'{name} Training Accuracy: {train_metrics["accuracy"]:.4f}')
        print(f'{name} Training Precision: {train_metrics["precision"]:.4f}')
        print(f'{name} Training Recall: {train_metrics["recall"]:.4f}')
        print(f'{name} Training F1-score: {train_metrics["f1"]:.4f}')

        test_metrics = evaluate_model(model, X_test, y_test)
        print(f'{name} Testing Accuracy: {test_metrics["accuracy"]:.4f}')
        print(f'{name} Testing Precision: {test_metrics["precision"]:.4f}')
        print(f'{name} Testing Recall: {test_metrics["recall"]:.4f}')
        print(f'{name} Testing F1-score: {test_metrics["f1"]:.4f}')

    # Section 8
    train_clean_series = pd.Series(train_clean, index=train_df.index)
    test_clean_series = pd.Series(test_clean, index=test_df.index)
    language_model_train = train_clean_series[train_clean_series != ""].sample(n=lm_train_size, replace=False, random_state=42).tolist()
    language_model_test = test_clean_series[test_clean_series != ""].sample(n=lm_test_size, replace=False, random_state=42).tolist()

    bigram_model = fit_bigram(language_model_train)

    results = {}
    for k in (1.0, 0.1, 0.01):
        results[k] = evaluate_bigram(bigram_model, language_model_test, k=k)

    print(f'Bigram vocabulary size: {len(bigram_model["vocab"])}')
    print(f'Bigram unseen test transitions: {results[1.0]["unseen"]}')
    print(f'Bigram test perplexity: {results[1.0]["perplexity"]:.4f}')
    print(f'Bigram test perplexity (k=0.1): {results[0.1]["perplexity"]:.4f}')
    print(f'Bigram test perplexity (k=0.01): {results[0.01]["perplexity"]:.4f}')


if __name__ == "__main__":
    run_experiment()
