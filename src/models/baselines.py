from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_vader_baseline(texts):
    analyzer = SentimentIntensityAnalyzer()
    logger.info(f"Running VADER on {len(texts)} texts ...")
    results = []
    for text in texts:
        if not isinstance(text, str):
            text = str(text) if text else ""
        scores = analyzer.polarity_scores(text)
        results.append({
            "text": text,
            "vader_neg": scores["neg"],
            "vader_neu": scores["neu"],
            "vader_pos": scores["pos"],
            "vader_compound": scores["compound"],
        })
    return pd.DataFrame(results)


def run_lda_baseline(texts, n_topics=5):
    valid_texts = [str(t) for t in texts if t and str(t).strip()]
    if not valid_texts:
        logger.warning("No valid texts for LDA.")
        return None, None
    logger.info(f"Running LDA on {len(valid_texts)} texts, {n_topics} topics ...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    tf = tf_vectorizer.fit_transform(valid_texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="online")
    lda.fit(tf)
    return lda, tf_vectorizer


def display_lda_topics(model, feature_names, no_top_words=10):
    if model is None or feature_names is None or len(feature_names) == 0:
        return
    for topic_idx, topic in enumerate(model.components_):
        top_words = " | ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        logger.info(f"LDA Topic {topic_idx}: {top_words}")
