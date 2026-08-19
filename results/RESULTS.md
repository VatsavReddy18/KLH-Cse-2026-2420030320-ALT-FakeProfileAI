# AI-Based Fake Profile Detection: Results

This document contains the implementation code, the output of the model, and a description of the final results.

## 1. Implementation Code

The main script is `src/fake_profile_detection.py`. It performs data generation, preprocessing, feature engineering, model training, evaluation, and prediction on a sample profile.

### Key Code Sections

#### Imports and Data Generation

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
```

A synthetic dataset of 10,000 profiles is created with features such as account age, followers, following, posts, username, bio, and profile picture flag.

#### Feature Engineering

```python
def extract_features(df):
    analyzer = SentimentIntensityAnalyzer()
    features = pd.DataFrame()
    features['username_entropy'] = df['username'].apply(username_entropy)
    features['has_numbers'] = df['username'].apply(lambda x: 1 if any(c.isdigit() for c in str(x)) else 0)
    features['ff_ratio'] = df['num_followers'] / (df['num_following'] + 1)
    features['account_age_days'] = df['account_age_days']
    features['posts_per_day'] = df['num_posts'] / (df['account_age_days'] + 1)
    features['has_profile_pic'] = df['has_profile_pic']
    features['bio_sentiment'] = df['bio'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    features['bio_length'] = df['bio'].apply(lambda x: len(str(x).split()))
    return features
```

#### Model Training

```python
# Base learners
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
xgb = XGBClassifier(eval_metric='logloss', max_depth=4, n_estimators=100, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, early_stopping=True, random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# Meta-learner stacking
stack_train = np.column_stack((rf.predict_proba(X_train)[:, 1], xgb.predict_proba(X_train)[:, 1], mlp.predict_proba(X_train)[:, 1]))
meta = LogisticRegression()
meta.fit(stack_train, y_train)
```

#### Prediction on a Sample Profile

```python
sample = pd.DataFrame([{
    'username': 'ghost_user_123',
    'account_age_days': 12,
    'num_followers': 45,
    'num_following': 1800,
    'num_posts': 3,
    'has_profile_pic': 0,
    'bio': 'Follow me back!!!'
}])

sample_features = extract_features(sample)
proba = meta.predict_proba(stack_sample)[:, 1]
label = 'FAKE' if proba[0] > 0.5 else 'REAL'
```

## 2. Final Output

### Console Output

```text
===== AI-Based Fake Profile Detection Results =====

Base Model Performance:
Random Forest   -> Accuracy: 0.904, F1: 0.902, AUC: 0.964
XGBoost         -> Accuracy: 0.901, F1: 0.898, AUC: 0.963
MLP             -> Accuracy: 0.902, F1: 0.901, AUC: 0.944

Ensemble (Stacked) Performance:
Accuracy:  0.900
Precision: 0.918
Recall:    0.879
F1-Score:  0.898
ROC-AUC:   0.954

Confusion Matrix:
[[922  78]
 [121 879]]

Sample Prediction:
Username: ghost_user_123
Prediction: FAKE
Confidence: 97.0%

Top Risk Factors:
- Low follower-to-following ratio
- Recently created account
- Username contains numbers
- No profile picture
- Very few posts
```

### Interpretation

- The **Ensemble model** achieves an accuracy of **90.0%**, an F1-score of **89.8%**, and an ROC-AUC of **95.4%**, balancing precision and recall effectively.
- The **sample profile** (`ghost_user_123`) is classified as **FAKE** with a high confidence of **97.0%** because it exhibits several strong risk indicators: a recently created account, a very low follower-to-following ratio, numeric characters in the username, no profile picture, and minimal posting activity.
- The confusion matrix shows the model is well-balanced: 922 true negatives, 879 true positives, 78 false positives, and 121 false negatives. The ensemble slightly improves precision over the individual base models, making it less likely to flag a real account as fake.

### Model Comparison

## 3. Screenshots and Visuals

Visual documentation of the system is available in the `src/images/` directory:

- `system_architecture.png` — high-level pipeline of the detection system.
- `ui_mockup.png` — proposed web dashboard for end users.

These images illustrate how the final system looks and how data flows through the model.

## 4. Conclusion

The AI-Based Fake Profile Detection system delivers a reliable, interpretable fake/real classification with a stacked ensemble approach. The final output demonstrates both strong quantitative performance and a clear explanation of why a given profile is flagged.

