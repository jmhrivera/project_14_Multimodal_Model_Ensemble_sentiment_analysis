# Project 16: Multimodal Model Ensemble for Sentiment Analysis

## Overview
Film Junky Union, a cutting-edge community for classic movie enthusiasts, is developing a system to filter and categorize movie reviews. Our objective is to train a model to automatically detect negative reviews. We'll use a dataset of movie reviews from IMDB with labels to build a model that classifies reviews as either positive or negative. The goal is to achieve an F1 score of at least 0.85.

## Instructions
1. Load the data.
2. Preprocess the data if necessary.
3. Perform exploratory data analysis and conclude on class imbalance.
4. Preprocess the data for modeling.
5. Train at least three different models on the training dataset.
6. Test the models on the test dataset.
7. Write some reviews and classify them with all models.
8. Compare the test results across models and attempt to explain the differences.
9. Present your findings.

## Provided Code Fragments
For your convenience, the project template already contains some code snippets which you can use or modify:
- Some exploratory data analysis with a few plots.
- `evaluate_model()`: A routine to evaluate a classification model that fits the scikit-learn prediction interface.
- `BERT_text_to_embeddings()`: A routine to convert a list of texts into embeddings using BERT.

## Requirements
- Python 3.x
- Pandas
- Numpy
- Scikit-learn
- Matplotlib/Seaborn (for exploratory data analysis)
- TensorFlow/PyTorch (if using BERT)
