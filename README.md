# Part 3: NLP – Customer Support Text Classification

An NLP pipeline that classifies customer support messages as Positive, Neutral, or Negative using TF-IDF + Logistic Regression (baseline) and an LSTM model.

## Dataset

- 1500 records, 6 columns
- Target: `sentiment_label` (positive / neutral / negative)
- Input: `customer_message` (avg ~13 words per message)
- Balanced classes: ~500 each

## Tasks Covered

1. Dataset exploration and class distribution
2. Text cleaning — lowercase, remove symbols, stopword removal
3. TF-IDF vectorization
4. Baseline model — Logistic Regression
5. Sequence model — LSTM with Embedding layer
6. Reflection on RNN, LSTM, Attention, and Transformers

## How to Run

pip install -r requirements.txt
jupyter notebook notebook.ipynb

Place `customer_support_text_classification.csv` in the same folder as the notebook.

## Model Results

| Model               | Test Accuracy |
|---------------------|---------------|
| Logistic Regression | ~85%          |
| LSTM                | ~82%          |

Actual values will appear after running the notebook.

## Key Concepts

**TF-IDF:** Converts text to numbers by scoring words based on how unique they are to a document vs. the whole dataset.

**LSTM:** A type of RNN with memory gates that can remember important context from earlier in a sequence.

**Attention:** Lets the model focus on the most relevant words at each step instead of compressing everything into one vector.

**Transformers:** Use self-attention in parallel across all words — the backbone of GPT, BERT, and all modern GenAI.

## Project Structure

notebook.ipynb
customer_support_text_classification.csv
requirements.txt
README.md
results/
├── model_evaluation.png
└── sample_predictions.txt
