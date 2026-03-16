## SafeType — AI-Powered Content Safety Detection for Children

# AIPI 540 Deep Learning Applications Project -2. 

We built a real-time content safety system (Android app and website) that monitors children's chat messages across Android apps (WhatsApp, Instagram, Snapchat, etc.) and classifies them into **10 categories**: clean, racism, sexism, profanity, cyberbullying, toxicity, hate speech, implicit hate, threat, and sarcasm.

The system includes a custom-trained DistilBERT model, a deployed API, an Android keyboard app that captures messages, and two web interfaces for monitoring and demonstration.

## Live Demos

- **Website and Parent Dashboard**: https://safetype.up.railway.app
- **Model API**: https://dl-project-2-second-version.onrender.com

## Project Structure

```
├── README.md               <- This file
├── requirements.txt        <- Python dependencies
├── setup.py                <- Runs full pipeline: data → features → train → experiments
├── main.py                 <- CLI inference (interactive, single text, or batch)
├── render.yaml             <- Render deployment config
├── scripts/
│   ├── make_dataset.py     <- Combines 8 data sources into unified 10-class dataset
│   ├── build_features.py   <- Text cleaning, TF-IDF, handcrafted features
│   ├── model.py            <- Trains Naive Baseline, Logistic Regression, DistilBERT
│   └── experiment.py       <- Training size sensitivity + noise robustness
├── models/
│   ├── naive_baseline.pkl
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── distilbert-toxicity/
├── data/
│   ├── raw/                <- Raw datasets (not tracked)
│   ├── processed/          <- Cleaned, feature-engineered data
│   └── outputs/            <- Results, plots, confusion matrices
├── notebooks/
│   └── eda.ipynb           <- Exploratory data analysis
├── web_app/                <- Deployed mobile app and web application code
│   ├── app.py              <- Flask API with post-processing calibration
│   ├── frontend/           <- React UI: iPhone keyboard simulator
│   ├── web/                <- Parent dashboard: real message monitoring
│   └── supabase/           <- Edge Function for batch analysis
└── .gitignore
```

## Quick Start

### Train models from scratch

```bash
pip install -r requirements.txt
python setup.py
```

This runs the full pipeline: data loading, feature engineering, model training, and experiments.

### Run inference locally

```bash
python main.py --model distilbert --text "you are so stupid"
```

### Run the API locally

```bash
cd web_app
python app.py
```

API endpoint: `POST /api/predict` with `{ "text": "...", "model": "distilbert" }`

## Models

| Model | Type | Accuracy | F1 |
|-------|------|----------|-----|
| Naive Baseline | Keyword matching | ~0.21 | ~0.17 |
| Logistic Regression | TF-IDF features | ~0.63 | ~0.62 |
| **DistilBERT** | Fine-tuned transformer | **~0.72** | **~0.72** |

All models classify into 10 categories. The API includes post-processing calibration that reduces false positives using severity-tiered confidence thresholds.

## Architecture

```
Android Device                    Cloud
┌─────────────┐         ┌─────────────────────┐
│ SafeType    │         │  Supabase           │
│ Keyboard    │────────>│  (messages DB)      │
│ + Screen    │         │         │           │
│   Reader    │         │    Edge Function    │
└─────────────┘         │         │           │
                        └─────────┼───────────┘
                                  │
                        ┌─────────▼───────────┐
                        │  Render API         │
                        │  Flask + DistilBERT │
                        │  /api/predict       │
                        └─────────┬───────────┘
                                  │
                        ┌─────────▼───────────┐
                        │  Railway            │
                        │  Parent Dashboard   │
                        │  (real-time alerts) │
                        └─────────────────────┘
```

- **Android app** captures messages via keyboard input, notifications, and screen reading
- **Supabase** stores messages and runs an Edge Function for batch analysis
- **Render** hosts the Flask API with the DistilBERT model
- **Railway** hosts the parent monitoring dashboard

## Data

8 datasets combined into ~375k samples, balanced to 50,000 (5,000 per class):

- Davidson et al. (24k tweets)
- Jigsaw Toxic Comment v1 (159k comments)
- Jigsaw Unintended Bias v2 (100k sampled)
- UC Berkeley Measuring Hate Speech (39k)
- EDOS Online Sexism (20k)
- HateXplain (20k)
- ImplicitHate (6k)
- TweetEval Irony (4.6k)

## Experiments

- **Training size sensitivity**: F1 grows from 0.36 (1% data) to 0.62 (100%)
- **Noise robustness**: F1 drops from 0.62 (clean) to 0.22 (30% character noise)

Results and plots in `data/outputs/`.
