# Fake News Classification

## Introduction
Fake news has become a major concern in recent years, especially on social media platforms. This project focuses on classifying COVID19-related news on Twitter as either fake or real using supervised machine learning techniques. We explore various binary classification algorithms, including Multinomial Naive Bayes, Logistic Regression, Passive Aggressive Classifier, Multi-Layer Perceptron, and Bidirectional Encoder Representations from Transformers (BERT). Additionally, we compare two natural language processing vectorization techniques: Bag of Words and Term Frequency-Inverse Document Frequency (TF-IDF).

The primary goal is to identify the most effective classification model for COVID19-related tweets. The project's findings have broader implications for combating fake news in society.

## Methodology
- Data Preprocessing: Cleaning and encoding data, including text cleaning (removing empty cells, punctuation, and HTML code) and label encoding.
- Tokenization: Converting text into tokens, with common terms and English stop words removed.
- Vectorization: Transforming text data into vectors using two methods: Bag of Words and TF-IDF.
- Models:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Passive Aggressive Classifier
  - Multi-Layer Perceptron
- BERT (Bidirectional Encoder Representations from Transformers): A sophisticated model designed to understand word context.

## Results
The table below summarizes the model accuracies:

| Model Name                 | Accuracy | Vectorizer |
| -------------------------- | -------- | ---------- |
| Multinomial Naive Bayes    | 0.89525  | TF-IDF     |
| Logistic Regression        | 0.91     | TF-IDF     |
| Multinomial Naive Bayes    | 0.913    | Bag of Words |
| Passive Aggressive Classifier | 0.91975 | Bag of Words |
| Logistic Regression        | 0.92375  | Bag of Words |
| Multi-Layer Perceptron     | 0.93025  | Bag of Words |
| Passive Aggressive Classifier | 0.934 | TF-IDF     |
| Multi-Layer Perceptron     | 0.939    | TF-IDF     |
| BERT                       | 0.9765   | BERT       |

According to the table, the BERT model achieved the highest accuracy of 0.9765, making it the most efficient among the tested models. The Multi-Layer Perceptron with TF-IDF vectorization followed with an accuracy of 0.939, and the Passive Aggressive Classifier with TF-IDF achieved an accuracy of 0.934.
