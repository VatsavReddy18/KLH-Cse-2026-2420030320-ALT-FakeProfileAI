"""
preprocessing.py

Data preprocessing for the AI-Based Fake Profile Detection project.
Cleans and prepares profile metadata, behavioural data, and network
graph data prior to feature engineering.
"""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load a raw dataset (Instagram / Twitter / generic fake-account dataset) from CSV."""
    return pd.read_csv(path)


def clean_profile_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Clean username, bio, profile picture, and account-age fields."""
    df = df.drop_duplicates()
    df = df.dropna(subset=["username"]) if "username" in df.columns else df
    return df


def clean_behavioural_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean posting frequency, activity timing, and engagement ratio fields."""
    return df.fillna(0)


def clean_graph_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean follower/following ratio and community/connection density fields."""
    return df.fillna(0)


if __name__ == "__main__":
    # Example usage — update paths once datasets are downloaded locally.
    raw = load_raw_data("data/raw/instagram_accounts.csv")
    cleaned = clean_profile_metadata(raw)
    cleaned.to_csv("data/processed/instagram_accounts_clean.csv", index=False)
