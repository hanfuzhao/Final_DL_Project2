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

- **Website & Parent Dashboard:** [safetype.up.railway.app](https://safetype.up.railway.app)
- **Model API & Demo Simulator:** [dl-project-2-second-version.onrender.com](https://dl-project-2-second-version.onrender.com)

## Project Structure

The repository is split into two layers: the **ML pipeline** (root-level scripts, models, data) and the **application layer** (`web_app/` — contains the Flask API, both web interfaces, the Android app, and Supabase backend).

```
├── README.md               <- Description of project and how to set up and run it
├── requirements.txt        <- Requirements file to document dependencies
├── setup.py                <- Script to set up project (get data, build features, train model)
├── main.py                 <- Main script to run project / user interface (CLI inference)
├── render.yaml             <- Render.com deployment configuration
├── scripts/                <- Directory for pipeline scripts
│   ├── make_dataset.py     <- Script to get data (combines 8 sources into 10-class dataset)
│   ├── build_features.py   <- Script to run pipeline to generate features (TF-IDF, handcrafted)
│   ├── model.py            <- Script to train model and predict (Naive, LR, DistilBERT)
│   └── experiment.py       <- Script to run experiments (training size, noise robustness)
├── models/                 <- Directory for trained models
│   ├── naive_baseline.pkl
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── distilbert-toxicity/
├── data/                   <- Directory for project data
│   ├── raw/                <- Directory for raw data (not tracked in git, ~3GB)
│   ├── processed/          <- Directory to store processed data
│   └── outputs/            <- Directory to store output data (results, plots, confusion matrices)
├── notebooks/              <- Directory to store exploration notebooks
│   └── eda.ipynb
├── web_app/                <- Application layer (API, web UIs, Android app, Supabase backend)
│   ├── app.py              <- Flask API server with post-processing calibration
│   ├── frontend/           <- React + Vite demo UI (iPhone keyboard simulator)
│   ├── web/                <- Parent monitoring dashboard (HTML/CSS/JS, Dockerfile, nginx)
│   ├── android_app/        <- Android app in Kotlin (keyboard, scrapers, background sync)
│   └── supabase/           <- Supabase Edge Function (batch analysis + email alerts)
└── .gitignore              <- Git ignore file
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
4. `experiment.py` — runs training size sensitivity and noise robustness experiments

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

**Data flow:** The Android app captures messages via three methods — keyboard input (IME), notification listening, and accessibility-based screen reading. Messages are stored locally in a Room database, deduplicated, and uploaded to Supabase via background WorkManager tasks. A Supabase Edge Function sends messages to the Flask API for classification, updates the database with flag results, and sends email alerts to parents via the Resend API when harmful content is detected. The parent dashboard on Railway displays flagged messages with severity levels, real-time alerts via Supabase Realtime subscriptions, and runs auto-analysis every 60 seconds.

## Models

| Model | Type | Accuracy | Weighted F1 |
|-------|------|----------|-------------|
| Naive Baseline | Keyword matching + majority class | ~0.21 | ~0.17 |
| Logistic Regression | TF-IDF (20k features, bigrams) + balanced LR | ~0.63 | ~0.62 |
| **DistilBERT** | Fine-tuned `distilbert-base-uncased` (3 epochs) | **~0.72** | **~0.72** |

All models classify into 10 categories. The deployed API includes **post-processing calibration** that reduces false positives using severity-tiered confidence thresholds (25% for severe categories like racism/threat, 50% for moderate, 75% for sarcasm).

## Data

8 datasets combined into ~375k samples, balanced to **50,000** (5,000 per class):

| Dataset | Samples | Primary Labels |
|---------|---------|----------------|
| Davidson et al. | ~24k | hate_speech, profanity, cyberbullying, clean |
| Jigsaw Toxic Comment v1 | ~159k | toxicity, profanity, cyberbullying, threat, racism, clean |
| Jigsaw Unintended Bias v2 | ~100k (sampled) | toxicity, clean |
| UC Berkeley D-Lab | ~39k | racism, sexism, hate_speech, toxicity, clean |
| EDOS SemEval 2023 | ~20k | sexism, threat, clean |
| HateXplain | ~20k | racism, sexism, hate_speech, profanity, clean |
| SALT-NLP ImplicitHate | ~6k | implicit_hate |
| TweetEval Irony | ~4.6k | sarcasm, clean |

**10 target classes:** `clean`, `racism`, `sexism`, `profanity`, `cyberbullying`, `toxicity`, `hate_speech`, `implicit_hate`, `threat`, `sarcasm`

## Experiments

Two experiments evaluate the classical ML model (Logistic Regression):

- **Training Size Sensitivity:** F1 score grows from 0.36 (1% of data, ~400 samples) to 0.62 (100% of data, ~40k samples)
- **Noise Robustness:** F1 drops from 0.62 (clean input) to 0.22 (30% character-level noise — random swaps, insertions, deletions)

Results, plots, and confusion matrices are saved in `data/outputs/`.

## Web Application

### Parent Dashboard (`web_app/web/`)

A full-featured monitoring dashboard deployed on Railway:

- Landing page with product overview, how-it-works steps, and FAQ
- Setup guide with step-by-step Android app installation instructions
- Parent login for authenticated dashboard access
- **Dashboard** with:
  - Stats strip: messages today, threats detected, apps monitored, last sync time
  - Device online/offline status indicator
  - Message list with filters by app (WhatsApp, Instagram, Snapchat, SMS, Keyboard), flag status, and capture source (accessibility, keyboard, notification)
  - Click any message for detailed 10-category probability breakdown
- Safety Guide with educational advice for parents and age-appropriate explanations for children
- Real-time sync via Supabase Realtime — new messages appear instantly
- Auto-analysis runs every 60 seconds on unanalyzed messages

### Demo Simulator (`web_app/frontend/`)

A React + Vite interactive demo deployed on Render alongside the API:

- iPhone-style chat interface with a fully functional on-screen keyboard
- Live typing detection — messages are analyzed in real-time as you type (350ms debounce)
- Send messages and see instant classification with probability bars
- Alerts panel and Safety Guide with per-category educational content

### Email Alerts (`web_app/supabase/`)

A Supabase Edge Function (Deno/TypeScript) that:

- Fetches unanalyzed messages from the database
- Sends each to the SafeType API for DistilBERT classification
- Updates the database with flag status and reason
- Sends an HTML email alert to the parent via the Resend API when any messages are flagged, including category, confidence score, message preview, severity color coding, and a direct link to the parent dashboard

## Android App (`web_app/android_app/`)

Built in Kotlin, the Android app runs on the child's device and captures messages through three layers:

- **SafeTypeIME** — Custom keyboard that captures all typed text across any app
- **NotificationCaptureService** — Listens to incoming notifications from messaging apps
- **ScreenScraperService** — Accessibility service that reads on-screen message content

Supporting components:

- App-specific scrapers for WhatsApp, Instagram, Snapchat, SMS, and a generic fallback
- Local Room database with deduplication engine to avoid duplicate captures
- Background sync via WorkManager (upload to Supabase + trigger analysis)
- Pairing flow for linking the child's device to the parent dashboard
- Boot persistence — monitoring restarts automatically after device reboot
- Transparency — persistent notification displays "SafeType Parental Monitor is active"
- Pre-built APKs — `Child_Side_app.apk` and `Safe_Type_Parent Side.apk` included

## Deployment

| Component | Platform | Details |
|-----------|----------|---------|
| Model API + Demo UI | Render | Flask + Gunicorn, CPU-only PyTorch, serves React demo + `/api/predict` |
| Parent Dashboard | Railway | Nginx + Docker, static HTML/CSS/JS, connects to Supabase |
| Message Database | Supabase | PostgreSQL with RLS, Realtime subscriptions, Edge Functions |
| Email Alerts | Supabase Edge Functions | Deno/TypeScript, sends HTML emails via Resend API |
| Android App | Sideloaded APK | Kotlin, requires IME + Accessibility + Notification permissions |
