# Source Code (`src/`)

This folder contains the runnable source code and visual assets for the AI-Based Fake Profile Detection project.

## Files

| File | Description |
|------|-------------|
| `fake_profile_detection.py` | Complete end-to-end script: data generation, feature engineering, model training, evaluation, and sample prediction. |
| `requirements.txt` | Python dependencies required to run the script. |
| `images/system_architecture.png` | High-level system pipeline diagram. |
| `images/ui_mockup.png` | Proposed web dashboard mockup. |

## Running the Code

```bash
# From the project root
pip install -r src/requirements.txt
python src/fake_profile_detection.py
```

## What the Script Does

1. **Generates synthetic data**: 10,000 fake and real social-media profiles with realistic distributions.
2. **Engineers features**: username entropy, numeric patterns, follower/following ratio, posting rate, bio sentiment, profile-picture flag, etc.
3. **Trains base models**: Random Forest, XGBoost, and a Multi-Layer Perceptron.
4. **Stacks predictions**: a Logistic Regression meta-learner combines the three base models.
5. **Evaluates**: accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
6. **Predicts**: runs a sample suspicious profile and prints a confidence score plus risk factors.

## Visuals

The images below document the project architecture and user interface.

### System Architecture

![System Architecture](images/system_architecture.png)

### Dashboard UI Mockup

![Dashboard UI](images/ui_mockup.png)

## Notes

- The dataset is synthetic, intended for demonstration and academic use.
- For real-world deployment, replace the synthetic data with labeled platform data and add adversarial robustness checks.
- The dashboard mockup is a design reference; a full frontend is not included in this repository.
