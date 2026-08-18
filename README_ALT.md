# AI-Based Fake Profile Detection in Social Networks

## Team Members

| S.No. | Name | Roll No. |
|-------|------|----------|
| 1 | D. Sree Aditya | 2420090056 |
| 2 | K. Rishikesh Reddy | 2420030318 |
| 3 | D. Srivatsava Reddy | 2420030320 |

**Supervisor:** _[Add supervisor name here]_
**Course:** Applied Machine Learning for Text Analysis (AI-driven Language and Technologies Specialization)
**Institution:** KL Hyderabad University (KLH)

## Abstract

The rapid growth of social networking platforms has led to a significant rise in fake, bot-operated, and impersonation profiles that spread misinformation, conduct scams, manipulate engagement metrics, and compromise user trust and platform integrity. Traditional rule-based detection methods rely on a limited set of hand-crafted heuristics and often fail to keep pace with increasingly sophisticated fake account behaviour. This project proposes an AI-Based Fake Profile Detection framework that leverages machine learning and deep learning techniques to automatically identify fake, bot, or impersonated profiles on social networks with high accuracy and reliability.

The proposed system analyses multiple categories of profile signals, including profile metadata (username patterns, profile picture authenticity, bio content, account age), behavioural patterns (posting frequency, activity timing, engagement ratios), and network-level graph features (follower-following ratios, community structure, connection density). Text-based attributes such as bio and post content are processed using NLP techniques and transformer-based embeddings (e.g., BERT), while structural attributes are modelled using graph-based methods such as Graph Neural Networks (GNNs). Classical machine learning models (Random Forest, SVM, Logistic Regression) and deep learning models (CNN, LSTM) are trained and benchmarked to identify the most effective detection approach.

The system performs data preprocessing, feature extraction, model training, and classification to flag profiles as genuine or fake/bot, along with a confidence score indicating detection certainty. To improve trust and transparency, an explainability layer using techniques such as SHAP, LIME, or feature-importance analysis is incorporated so that the specific attributes driving a prediction can be understood by platform moderators and end users. Model performance is evaluated using accuracy, precision, recall, F1-score, and ROC-AUC, while explainability outputs are assessed for consistency and interpretability. The resulting system aims to provide accurate, scalable, and explainable fake profile detection that supports content moderation, platform security, and safer user experiences on social networks.

## Datasets

| Dataset | Link |
|---------|------|
| Fake and Genuine Profile Dataset (Instagram) | https://www.kaggle.com/datasets/free4ever1/instagram-fakespammer-genuine-accounts |
| Fake Social Media Account Detection Dataset | https://www.kaggle.com/datasets/satishkumarprajapati/fakesocial-media-account-detection-dataset |
| Genuine/Fake User Profile Dataset (Twitter) | https://www.kaggle.com/datasets/whoseaspects/genuinefakeuser-profile-dataset |

## Approach

1. **Data Collection** — Profile metadata, behavioural data, and network graph data from multiple sources.
2. **Data Preprocessing** — Cleaning and structuring raw profile, behavioural, and graph data.
3. **Feature Engineering** — Extracting text, behavioural, and graph-based features.
4. **Model Training / Fine-tuning** — Random Forest, SVM, CNN/LSTM, GNN, and BERT-based models.
5. **Fake Profile Classifier** — Combines model outputs to classify profiles as genuine or fake/bot.
6. **Confidence & Risk Scoring** — Attaches a confidence score to each prediction.
7. **Explainability Layer** — SHAP, LIME, and feature-importance analysis for interpretable predictions.
8. **Flagged Profile Insights & Alerts** — Actionable, human-readable outputs.
9. **Web Dashboard** — Presents flagged profiles and insights to moderators/end users.

## Repository Structure

```
├── src/          # Source code (preprocessing, feature engineering, models, dashboard)
├── docs/         # Documentation, design notes, references
├── data/         # Data or documented data source references (see Datasets section)
├── results/      # Model outputs, evaluation metrics, plots
├── reports/      # Phase reports and review submissions
└── README.md
```

## Setup and Execution Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the datasets listed above and place them under `/data`, or update the data source reference in `/data/README.md`.
5. Run preprocessing and feature engineering scripts:
   ```bash
   python src/preprocessing.py
   python src/feature_engineering.py
   ```
6. Train and evaluate models:
   ```bash
   python src/train.py
   ```
7. Launch the dashboard (once implemented):
   ```bash
   python src/dashboard.py
   ```

## Current Phase Status

_[Update this section each phase — e.g., "Phase 1: Literature review and dataset finalization complete. Phase 2: Data preprocessing and feature engineering in progress."]_

## Evaluation Metrics

Accuracy, Precision, Recall, F1-score, and ROC-AUC are used to evaluate model performance. Explainability outputs (SHAP/LIME) are assessed for consistency and interpretability.

## Notes

- No credentials, API keys, licensed datasets, or confidential institutional data are stored in this repository.
- Each team member commits under their own GitHub account.
- Phase deliverables are tagged (`review-1`, `review-2`, `final`) as per submission norms.
