# Part 3 — NLP and Sequence Modeling: Customer Support Sentiment Classification

## Problem Statement
Text classification task to predict the sentiment (positive, neutral,
negative) of customer support tickets submitted across multiple channels.

## Dataset
- **Source:** Provided synthetic dataset
- **File:** customer_support_text_classification.csv
- **Records:** 1500 tickets | 3 sentiment classes
- **Features:** customer_message (text), channel, word_count, urgent_flag
- **Target:** sentiment_label (positive / neutral / negative)
- **Dataset Link:** https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs

## Approach
- Cleaned text: lowercasing, removing special characters, stopword removal
- Vectorized using TF-IDF with bigrams for baseline model
- Built Logistic Regression baseline (TF-IDF features)
- Built LSTM sequence model with Embedding layer
- Compared both models on test accuracy and per-class F1

## Results
See results/ folder for model evaluation outputs and sample predictions.

## Repository Structure
part-3-nlp-sequence-modeling/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_evaluation.png
    └── sample_predictions.txt
