# System Architecture

## Pipeline Overview

```
 MULTIPLE DATA SOURCES
    ┌────────────────┼────────────────┐
    ↓                ↓                ↓
 Profile Metadata  Behavioural Data  Network Graph
 (bio, images,     (posting freq,    (followers/
 username)         likes, activity)  following ratio)
    │                │                │
    └────────────────┼────────────────┘
                      ↓
              Data Preprocessing
                      ↓
              Feature Engineering
        (text, behavioural, graph features)
                      ↓
            Model Training / Fine-tuning
      (Random Forest, SVM, CNN/LSTM, GNN, BERT)
                      ↓
              Fake Profile Classifier
                      ↓
            Confidence & Risk Scoring
                      ↓
              Explainability Layer
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
       SHAP             LIME       Feature Importance
        └────────────────┼────────────────┘
                          ↓
              Human-Readable Explanation
                          ↓
              Flagged Profile Insights
                          ↓
              Actionable Insights / Alerts
                          ↓
                   Web Dashboard
```

## Components

- **Data Sources**: Profile metadata, behavioural signals, and network graph data drawn from the datasets listed in `data/README.md` (or equivalent data source reference).
- **Preprocessing**: Cleaning, deduplication, and normalization (`src/preprocessing.py`).
- **Feature Engineering**: Text embeddings (BERT), behavioural ratios, and graph features (`src/feature_engineering.py`).
- **Models**: Classical ML (Random Forest, SVM, Logistic Regression) and deep learning (CNN, LSTM, GNN) benchmarked against each other (`src/train.py`).
- **Explainability**: SHAP, LIME, and feature-importance analysis for transparent predictions (`src/explainability.py`).
- **Dashboard**: Web interface surfacing flagged profiles, confidence scores, and explanations to moderators.

## Evaluation

Models are compared using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
