from typing import Optional
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_bertopic_clustering(docs: list, timestamps: Optional[list] = None):
    valid_docs = [str(d) for d in docs if d and str(d).strip()]
    if not valid_docs:
        logger.warning("No valid documents for BERTopic.")
        return None, None
    logger.info(f"BERTopic: clustering {len(valid_docs)} documents ...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model      = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model   = HDBSCAN(min_cluster_size=15, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    topic_model     = BERTopic(
        embedding_model       = embedding_model,
        umap_model            = umap_model,
        hdbscan_model         = hdbscan_model,
        language              = "english",
        calculate_probabilities = False,
    )
    topics, _ = topic_model.fit_transform(valid_docs)
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1
    logger.info(f"BERTopic discovered {n_topics} topics (-1 outliers excluded).")
    return topic_model, topic_info


def extract_dynamic_topics(topic_model, docs: list, timestamps: list):
    if topic_model is None or not timestamps:
        return None
    logger.info("Calculating dynamic topics over time ...")
    return topic_model.topics_over_time(docs, timestamps)
