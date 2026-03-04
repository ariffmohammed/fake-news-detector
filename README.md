# Fake News Detector

A machine learning project that classifies news headlines as real or fake using Python. Compares two models — Logistic Regression and Naive Bayes — on the same dataset.

## How It Works

News headlines are converted to numerical features using TF-IDF vectorization. Both Logistic Regression and Naive Bayes then learn the patterns between word importance scores and whether a headline is real or fake. Both predictions are shown side by side for comparison.

## What I Learned

- Naive Bayes classifier and how it differs from Logistic Regression
- Comparing multiple models on the same dataset
- Naive Bayes works well with small text datasets, Logistic Regression tends to be more accurate with larger data

## Limitations

- Small hardcoded dataset of 20 headlines
- Full version using a large real world fake news dataset coming soon

## Tech Stack

- Python
- scikit-learn

## Dataset

Sample headlines hardcoded for demonstration. Full project will use a real world fake news dataset with thousands of labeled articles.
