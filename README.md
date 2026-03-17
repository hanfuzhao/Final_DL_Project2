# SafeType — AI-Powered Content Safety Detection for Children

**AIPI 540 - Deep Learning Applications | Project 2**

We built a real-time content safety system that monitors children's chat messages across Android apps (WhatsApp, Instagram, Snapchat, SMS, etc.) and classifies them into **10 categories**: clean, racism, sexism, profanity, cyberbullying, toxicity, hate speech, implicit hate, threat, and sarcasm.

The system includes a **custom fine-tuned DistilBERT model**, a **deployed Flask API**, an **Android keyboard/monitor app** that captures messages, a **parent monitoring dashboard** with real-time alerts and email notifications, and an **interactive demo simulator**.

## Team Members

- **Jaideep**
- **Hanfu**
- **Keming**

> Parts of the Android app and web application were developed with assistance from Claude and Cursor.

## Live Demos

- **Model API & Demo Simulator:** [dl-project-2-second-version.onrender.com](https://dl-project-2-second-version.onrender.com)
- **Parent Dashboard:** [safetype.up.railway.app](https://safetype.up.railway.app)
  - Username: `username`
  - Password: `password`

## Project Structure

```
├── README.md
├── requirements.txt
├── setup.py                <- Runs full pipeline: data → features → train → experiments
├── main.py                 <- CLI inference (interactive, single text, or batch)
├── render.yaml             <- Render deployment config
├── scripts/
│   ├── make_dataset.py     <- Combines 8 data sources into unified 10-class dataset
│   ├── build_features.py   <- Text cleaning, TF-IDF, handcrafted features
│   ├── model.py            <- Trains Naive Baseline, Logistic Regression, DistilBERT
│   └── experiment.py       <- Training size sensitivity + noise robustness + ablation
├── models/
│   ├── naive_baseline/
│   ├── logistic_regression/
│   └── distilbert/
├── data/
│   ├── raw/                <- Raw datasets (not tracked, see download links below)
│   ├── processed/          <- Cleaned, feature-engineered data
│   └── outputs/            <- Results, plots, confusion matrices
├── notebooks/
│   └── eda.ipynb
├── web_app/
│   ├── app.py              <- Flask API with post-processing calibration
│   ├── frontend/           <- React UI: iPhone keyboard simulator
│   ├── web/                <- Parent dashboard: real message monitoring
│   ├── android_app/        <- Android app in Kotlin (keyboard, scrapers, sync)
│   └── supabase/           <- Edge Function for batch analysis + email alerts
└── .gitignore
```

## Quick Start

### Train models from scratch

```bash
pip install -r requirements.txt
python setup.py
```

This runs the full pipeline in order:

1. `make_dataset.py` — loads 8 raw datasets, maps to 10 classes, balances to 50k samples
2. `build_features.py` — cleans text, builds TF-IDF features, extracts handcrafted features
3. `model.py` — trains Naive Baseline, Logistic Regression, and fine-tunes DistilBERT
4. `experiment.py` — runs training size sensitivity, noise robustness, and ablation experiments

### Run inference locally

```bash
# Interactive mode
python main.py

# Single text prediction
python main.py --model distilbert --text "you are so stupid"

# Batch file prediction
python main.py --batch input.txt --model logistic_regression
```

Available models: `naive_baseline`, `logistic_regression`, `distilbert`

### Run the API locally

```bash
cd web_app
python app.py
```

API endpoint: `POST /api/predict` with JSON body `{ "text": "...", "model": "distilbert" }`

## Models

| Model | Type | Accuracy | Weighted F1 |
|-------|------|----------|-------------|
| Naive Baseline | Keyword matching + majority class | ~0.21 | ~0.17 |
| Logistic Regression | TF-IDF (20k features, bigrams) + balanced LR | ~0.63 | ~0.62 |
| **DistilBERT** | Fine-tuned `distilbert-base-uncased` (3 epochs) | **~0.72** | **~0.72** |

All models classify into 10 categories. The deployed API includes post-processing calibration that reduces false positives using severity-tiered confidence thresholds.

## Architecture

```
Child's Android Device               Cloud Infrastructure
┌──────────────────────┐     ┌──────────────────────────────┐
│  SafeType App        │     │  Supabase                    │
│  ├─ Custom Keyboard  │────>│  ├─ PostgreSQL (messages DB) │
│  ├─ Screen Scraper   │     │  └─ Edge Function            │
│  ├─ Notification     │     │      ├─ Batch analysis       │
│  │   Listener        │     │      └─ Email alerts (Resend)│
│  └─ Local Room DB    │     └──────────┬───────────────────┘
│     + WorkManager    │                │
└──────────────────────┘     ┌──────────▼───────────────────┐
                             │  Render                      │
                             │  Flask API + DistilBERT      │
                             │  POST /api/predict           │
                             └──────────┬───────────────────┘
                                        │
                             ┌──────────▼───────────────────┐
                             │  Railway                     │
                             │  Parent Dashboard            │
                             │  (real-time alerts & history)│
                             └──────────────────────────────┘
```

## Data

8 datasets combined into ~375k samples, balanced to **50,000** (5,000 per class):

| Dataset | Samples | Primary Labels | Download |
|---------|---------|----------------|----------|
| Davidson et al. | ~24k | hate_speech, profanity, cyberbullying, clean | [GitHub](https://github.com/t-davidson/hate-speech-and-offensive-language) |
| Jigsaw Toxic Comment v1 | ~159k | toxicity, profanity, cyberbullying, threat, racism, clean | [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) |
| Jigsaw Unintended Bias v2 | ~100k (sampled) | toxicity, clean | [Kaggle](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) |
| UC Berkeley D-Lab | ~39k | racism, sexism, hate_speech, toxicity, clean | [HuggingFace](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) |
| EDOS SemEval 2023 | ~20k | sexism, threat, clean | [GitHub](https://github.com/rewire-online/edos) |
| HateXplain | ~20k | racism, sexism, hate_speech, profanity, clean | [GitHub](https://github.com/punyajoy/HateXplain) |
| SALT-NLP ImplicitHate | ~6k | implicit_hate | [HuggingFace](https://huggingface.co/datasets/SALT-NLP/ImplicitHate) |
| TweetEval Irony | ~4.6k | sarcasm, clean | [HuggingFace](https://huggingface.co/datasets/tweet_eval) |

**10 target classes:** `clean`, `racism`, `sexism`, `profanity`, `cyberbullying`, `toxicity`, `hate_speech`, `implicit_hate`, `threat`, `sarcasm`

## Experiments

- **Training size sensitivity**: F1 grows from 0.36 (1% data) to 0.62 (100%)
- **Noise robustness**: F1 drops from 0.62 (clean) to 0.22 (30% character noise)
- **Ablation (preprocessing)**: Lemmatization hurts F1 from 0.632 to 0.609. Bigrams and sublinear TF scaling recover performance to 0.622.

Results, plots, and confusion matrices are saved in `data/outputs/`.

## AI Tools Used

| File | Tool | What it helped with |
|------|------|---------------------|
| `scripts/model.py` | Cursor | DistilBERT training loop boilerplate, confusion matrix plotting |
| `scripts/make_dataset.py` | ChatGPT | Dataset loading helpers, label mapping logic |
| `main.py` | Cursor | Argparse CLI setup |
| `web_app/app.py` | Cursor | CORS preflight handling, calibration thresholds |
| `web_app/web/js/app.js` | Cursor | Routing logic, event wiring |
| `web_app/frontend/src/App.jsx` | Claude | React component structure, keyboard UI layout |
| `web_app/android_app/` | Claude + Cursor | Android keyboard service, screen scraper, notification listener |
| Technical report | ChatGPT | Drafting, editing, and proofreading |

**Report:** [SafeType Technical Report](https://docs.google.com/document/d/1MksImt4q3zGvLqXKH9gqE7oQhb-_jC9q21UgQqF2ocA/edit?tab=t.0)

**Slides:** [SafeType Presentation](https://docs.google.com/presentation/d/1XSWjBVttds5of6nGg83aZBhHsgI7IQv86L3y4Kc7UaQ/edit)

## References

- Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated hate speech detection and the problem of offensive language. *ICWSM*.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS Workshop*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
- Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). The risk of racial bias in hate speech detection. *ACL*.
- Kennedy, C. J., Bacon, G., Sahn, A., & von Vacano, C. (2020). Constructing interval variables via faceted Rasch measurement and multitask deep learning: A hate speech application. *arXiv*.
- Mozafari, M., Farahbakhsh, R., & Crespi, N. (2020). A BERT-based transfer learning approach for hate speech detection in online social media. *Complex Networks*.
- Barbieri, F., Camacho-Collados, J., Neves, L., & Espinosa-Anke, L. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. *Findings of EMNLP*.
- Mathew, B., Saha, P., Yimam, S. M., et al. (2021). HateXplain: A benchmark dataset for explainable hate speech detection. *AAAI*.
- ElSherief, M., Ziems, C., Muchlinski, D., et al. (2021). Latent hatred: A benchmark for understanding implicit hate speech. *EMNLP*.
- Fortuna, P., & Nunes, S. (2018). A survey on automatic detection of hate speech in text. *ACM Computing Surveys*, 51(4), 1–30.
- Jigsaw/Google. Toxic Comment Classification Challenge. *Kaggle*.

## Deployment

| Component | Platform | Details |
|-----------|----------|---------|
| Model API + Demo UI | Render | Flask + Gunicorn, CPU-only PyTorch, serves React demo + `/api/predict` |
| Parent Dashboard | Railway | Static HTML/CSS/JS, connects to Supabase |
| Message Database | Supabase | PostgreSQL with RLS, Realtime subscriptions, Edge Functions |
| Email Alerts | Supabase Edge Functions | Deno/TypeScript, sends HTML emails via Resend API |
| Android App | Sideloaded APK | Kotlin, requires IME + Accessibility + Notification permissions |
