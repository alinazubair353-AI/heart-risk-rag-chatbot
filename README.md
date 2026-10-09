# Offline RAG Chatbot for Heart-Disease Risk

A fully local chatbot that reads lab reports (PDF, scanned image, CSV) and a typed lifestyle description, then explains cardiovascular risk **with citations**. Nothing leaves your computer: the language model runs through [Ollama](https://ollama.com), embeddings run locally, and the app only talks to `localhost`.

> **Educational prototype. Not a medical device and not medical advice.** Built as a university assignment (KRL course, Assignment 1).

## What it does

It combines symbolic reasoning, machine learning and retrieval, and shows how each contributed to an answer:

- **Ontology + knowledge graph:** OWL ontology with 10 object properties, 9 data properties and 13 defined classes. 78 sourced facts (647 RDF triples) and 6 SPARQL queries.
- **Extraction:** reads PDF, scanned reports (Tesseract OCR) and CSV, normalises units (e.g. mmol/L to mg/dL).
- **Rule engine:** 36 clinical rules with a readable trace and red-flag symptom detection.
- **Machine learning:** logistic regression, random forest and XGBoost, tuned, calibrated and explained with SHAP, plus a hybrid experiment (rule features added to the model).
- **RAG:** guideline documents are chunked (300 tokens, 60 overlap), embedded with `all-MiniLM-L6-v2`, indexed in FAISS, and retrieved with hybrid vector + keyword + graph search. `llama3.2:3b` writes the answer from the retrieved evidence.
- **Safety:** red-flag symptoms bypass the model with an urgent message, refuses medicine and dose questions, refuses out-of-scope questions, checks citations against retrieved passages, and always appends a disclaimer.
- **App:** FastAPI backend and a Streamlit interface (Chat, Extracted values, Risk dashboard, Rules & trace, Retrieved sources, Knowledge graph, Learning, Evaluation, Safety & ethics).

## How a question is answered

```
question
  -> safety checks (red flag / medicine request / out of scope)
  -> extraction of values into a patient record
  -> rules engine (trace) + calibrated ML risk (SHAP)
  -> hybrid retrieval of guideline passages (FAISS + keywords + graph)
  -> prompt with evidence -> local LLM (Ollama)
  -> citation check -> disclaimer
```

If Ollama is not running, the app falls back to an **extractive mode** that quotes the retrieved evidence. It is clearly labelled and is not a language-model answer.

## Quick start

Requirements: Python 3.10+ (developed on 3.12), [Ollama](https://ollama.com), and [Tesseract OCR](https://github.com/UB-Mannheim/tesseract) for scanned reports.

```bash
python -m venv venv
venv\Scripts\activate                 # Windows. Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

# one-time, needs internet: embedding model + local LLM
pip install sentence-transformers
python scripts/download_assets.py     # saves models/embedding and pulls llama3.2:3b

# build ontology, graph, models and index
python scripts/bootstrap.py --skip-eval

# run (two terminals)
run_backend.bat                       # http://localhost:8000/docs
run_frontend.bat                      # http://localhost:8501

pytest -q                             # tests
```

On Windows, if Tesseract is not found, set `TESSERACT_CMD` to the path of `tesseract.exe`.

## Datasets

Datasets are **not included**. Download them and place the CSV files in `data/datasets/`:

- UCI Heart Disease (Cleveland), 303 rows: https://archive.ics.uci.edu/dataset/45/heart-disease
- Kaggle Heart Failure Prediction, 918 rows: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Then train with `python -m backend.ml.train --full`. Check each dataset's licence before reuse. Note that the Kaggle dataset combines several UCI sources, so the two datasets overlap.

## Results so far

| Dataset | Best model | AUC | Recall |
|---|---|---|---|
| UCI Cleveland | Random forest | 0.740 | 0.571 |
| Kaggle Heart Failure | XGBoost | 0.797 | 0.810 |

These come from small held-out test sets (about 60 rows for Cleveland), so treat them as rough. The model uses only five features that exist in a patient record: age, sex, systolic blood pressure, total cholesterol and a fasting-glucose flag. Calibration changed the Brier score by about 0.003 or less, so the gain is small.

## Limitations

- The guideline files in `data/guidelines/` are **paraphrased study notes**, not the official WHO / AHA-ACC / ESC / ADA documents. Replace them with the official PDFs and re-run `python -m backend.rag.ingest` before relying on any citation.
- A red-flag symptom stays active for the rest of the session, so later normal questions in that session also get the urgent message. Start a new session to reset.
- A 3B local model can occasionally over-refuse or miss a detail from the question.
- The ML model is trained on small public datasets and is not clinically validated.

## Project structure

```
backend/        FastAPI app, extraction, knowledge (OWL, triples, SPARQL), reasoning (rules), ml, rag, safety
frontend/       Streamlit app
data/           guidelines, datasets (not included), synthetic test reports, evaluation profiles
evaluation/     extraction accuracy, retrieval and ablation runs, faithfulness scoring
scripts/        bootstrap, asset download, synthetic report generator
tests/          54 pytest tests
```

## AI assistance

I used Claude as a coding and learning assistant while building this project, and I tested and checked the results myself.

## Author

Your Name | [LinkedIn](https://www.linkedin.com/in/your-profile)
