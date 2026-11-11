# Email Classifier

A Python web demo that assigns pasted email text a category (Work, Personal, Spam) and priority (High, Medium, Low). Built from an existing personal learning project, with Codex assistance on repairs, validation, evaluation, and documentation.

## Problem and approach

Manually sorting messages is repetitive. This project explores text classification through a browser interface and a JSON API. TF-IDF converts words into numeric features; Multinomial Naive Bayes predicts category, and a three-neighbor classifier predicts priority. The browser displays results and server validation errors.

This is an educational prototype, not a reliable inbox assistant. It does not connect to Gmail, send messages, or store submitted text. Use fictional examples when demonstrating it.

## Run locally (Windows, Python 3.12)

Open PowerShell in this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000 and try `Team meeting tomorrow`. Stop with Ctrl+C. The page must be opened through this address rather than by double-clicking its HTML file.

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe evaluate.py
```

Tests cover the page, classification responses, invalid inputs, request limits, unknown vocabulary, malformed datasets, and repeatable evaluation. All five test methods pass in a fresh Python 3.12 environment.

## Measured results and limitations

The included dataset contains only 30 fictional, manually labeled messages. Each target is evaluated with a separate stratified 24/6 train/test split and seed 42. TF-IDF is fitted on training examples only. The browser models are then fitted on all 30 examples for demonstration.

| Target | Test accuracy | Macro F1 | Majority baseline accuracy |
| --- | --- | --- | --- |
| Category | 50.0% (3/6) | 0.389 | 33.3% |
| Priority | 83.3% (5/6) | 0.841 | 50.0% |

These six-example tests are too small to establish real-world performance. Priority labels are subjective; all Spam examples are labeled Low, which creates a strong artificial relationship. Category scores displayed in the UI are uncalibrated model probabilities, not a guarantee of correctness. Unknown words produce a helpful error, but familiar words can still yield wrong predictions. See `evaluation.json` for the recorded results and `data/README.md` for provenance.

## API

POST `/api/classify` with JSON `{"email_text":"Team meeting tomorrow"}` returns `category`, `priority`, and `confidence`. GET `/api/health` reports readiness. Non-object JSON, blank or non-string text, text over 10,000 characters, requests over 64 KiB, and entirely unrecognized vocabulary are rejected. The app does not log email bodies; ordinary Flask request logs contain request paths and statuses.

## Project files

- `app.py`: Flask app factory, routes, and request validation.
- `models/email_classifier.py`: dataset checks, classifiers, and evaluation.
- `templates/index.html`: browser interface.
- `data/emails.csv`: fictional example messages.
- `test_email_classifier.py`: automated checks.
- `evaluate.py`: reproducible evaluation command.

## Changes from the earlier learning version

Removed a malformed dataset row that prevented training; replaced the original examples with explicitly fictional, documented examples; separated held-out evaluation from serving; corrected misleading accuracy labeling; added request checks, error handling, project-relative paths, and tests. No improvement claim is made against the old dataset because the examples changed.

## Dependencies and attribution

Uses Flask, pandas, NumPy, and scikit-learn installed from PyPI; no paid services, API keys, copied pretrained models, or private inbox data are required. Their licenses remain with their respective projects. This repository does not bundle their source or virtual environments. Existing application structure and interface were retained from the local project supplied by AbbasRizvi-NEU; earlier third-party provenance, if any, has not been established. No new blanket license is assigned to that existing code.

## Next meaningful improvement

Obtain a larger appropriately licensed dataset with documented labels, evaluate across multiple splits, inspect mistakes, and assess probability calibration before making stronger accuracy claims.
