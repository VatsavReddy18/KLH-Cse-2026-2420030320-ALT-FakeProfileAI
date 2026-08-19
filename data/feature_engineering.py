"""
feature_engineering.py

Feature extraction for the AI-Based Fake Profile Detection project.
Produces text (NLP/BERT), behavioural, and graph-based features
consumed by the classical ML, deep learning, and GNN models.
"""

import pandas as pd


def extract_text_features(df: pd.DataFrame, text_col: str = "bio") -> pd.DataFrame:
    """
    Generate transformer-based embeddings (e.g., BERT) for bio/post text.
    Placeholder — integrate a sentence-transformers or HuggingFace model here.
    """
    df[f"{text_col}_length"] = df[text_col].astype(str).apply(len)
    return df


def extract_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive posting frequency, activity timing, and engagement ratio features."""
    if {"posts", "account_age_days"}.issubset(df.columns):
        df["posting_frequency"] = df["posts"] / df["account_age_days"].replace(0, 1)
    return df


def extract_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive follower-following ratio and community/connection density features."""
    if {"followers", "following"}.issubset(df.columns):
        df["follower_following_ratio"] = df["followers"] / df["following"].replace(0, 1)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/instagram_accounts_clean.csv")
    df = extract_text_features(df)
    df = extract_behavioural_features(df)
    df = extract_graph_features(df)
    df.to_csv("data/processed/instagram_accounts_features.csv", index=False)
