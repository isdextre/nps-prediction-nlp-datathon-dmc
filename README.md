# NPS Prediction — NLP Datathon DMC

Multiclass prediction of a university's student **Net Promoter Score (NPS)** category, combining structured survey fields with NLP-derived features extracted from open-text comments.

## Problem

The survey asked students *"Would you recommend the university to friends/acquaintances?"* on a 0–10 scale, bucketed into 4 ordinal classes:

| Score range | Class |
|---|---|
| 0–3 | 1 — Would not recommend |
| 4–5 | 2 — Unlikely to recommend |
| 6–8 | 3 — Likely to recommend |
| 9–10 | 4 — Would recommend |

The goal is to predict this class from respondent data and their free-text comment, so the university can identify the drivers behind satisfaction and prioritize where to act.

## Data

- `train_universidad.csv` — 16,000 labeled respondents.
- `test_universidad.csv` — 4,000 respondents for the datathon submission.
- Key columns: `COD_ENCUESTADO` (respondent id), `Nombre Campus`, `NIVEL ACTUAL` (mode: ON LINE / PRESENCIAL / FC / AC), `Clave de carrera` (program code), `Ciclo` (term), `COMENTARIO` (open-text comment, Spanish), `IND_GEA`, `IND_DELEGADO`, `CANT_CURSOS_MATRICU_SIN_INGLES`, `UOD_depostista_ind_deportista` (athlete flag), and the target `NPS` (1–4).
- `COMENTARIO` contains encoding artifacts (e.g. `Õ`, `Ð` in place of accented characters) from being read as `latin1` — handled during preprocessing.

## Approach (`Notebook.ipynb`)

**1. Cleaning & structured features** — binary indicators (GEA / class delegate / athlete) imputed from missing values, campus-mode encoding, missing-value handling.

**2. NLP feature engineering on `COMENTARIO`** — a large custom feature set built on top of the raw comment:

- TF-IDF (1–3 grams) + TruncatedSVD → 20 latent "component" features summarizing each comment.
- LDA topic modeling (20 topics) → per-comment topic-membership probabilities (`Tema_0`…`Tema_19`).
- Surface text statistics: word/letter counts, uppercase ratio, sentence count, average word length, lexical diversity, unique-word density.
- Spelling-error count (`pyspellchecker`, Spanish dictionary) and stop-word ratio.
- Rule-based lexicons: profanity flag, positive/negative phrase counts, keyword flags (e.g. "recomendaría", "excelente"), bigram/trigram frequency.
- Sentiment and subjectivity via `TextBlob`, plus a transformer sentiment score (`nlptown/bert-base-multilingual-uncased-sentiment`) with star rating and confidence.

**3. Feature selection** — Random Forest feature importance to shortlist the ~50 most informative variables (mostly SVD components, topic weights, sentiment score, and program/campus fields), with a correlation matrix as a sanity check.

**4. Modeling** — two strategies compared:

- **Multiclass baseline**: Random Forest and XGBoost, both tuned with `GridSearchCV`, trained directly on the 4-class target.
- **One-vs-rest neural networks**: the target is split into 4 binary problems (class *k* vs. rest), each corrected for class imbalance with random over/under-sampling, and fit with a separate feed-forward network (Keras, 128→64→32→2, dropout, softmax) per class; the four class probabilities are then recombined into the final submission. ELM (Extreme Learning Machine) and RBF classifiers were also tried per class as lighter alternatives to the neural net.

## Results

| Model | Accuracy | Log loss |
|---|---|---|
| Random Forest (baseline) | 0.686 | – |
| Random Forest (`GridSearchCV`-tuned) | 0.681 | 0.811 |
| XGBoost (`GridSearchCV`-tuned) | 0.660 | 0.876 |

Both baselines classify the majority classes (3 and 4, "likely"/"would recommend", ~80% of respondents) well (F1 0.70–0.84) but struggle on the minority classes 1 and 2 — the expected effect of class imbalance, which is what the one-vs-rest neural-network stage was designed to address.

## Repository contents

| File | Description |
|---|---|
| `Notebook.ipynb` | Full pipeline: EDA, feature engineering, feature selection, and model training/evaluation. |
| `train_universidad.csv` | Labeled training set (16,000 respondents). |
| `test_universidad.csv` | Unlabeled test set (4,000 respondents) for the datathon submission. |

## Running it

The notebook was built for Google Colab (paths like `/content/train_universidad.csv`). To run it elsewhere, place both CSVs alongside the notebook and update those paths, or upload them to a Colab session.

Key dependencies: `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `tensorflow`/`keras`, `nltk`, `textblob`, `pyspellchecker`, `stop-words`, `rippletagger`, `transformers`, `hpelm`.

One shortcut cell reads `train_con_sentimiento.csv` / `test_con_sentimiento.csv` to skip re-running the transformer sentiment model — those files aren't included in this repo, so re-run the sentiment-pipeline cells above it (or regenerate the files) if starting from a clean checkout.
