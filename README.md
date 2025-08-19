# Fake News Detection

A machine learning classifier that distinguishes between fake and real news articles using Natural Language Processing techniques. Built with TF-IDF vectorization and Linear Support Vector Classification, achieving 94.5% accuracy on news article classification.

## Author
**Macha Praveen**

## Overview

This project implements a binary text classification system designed to identify fake news articles. Using a dataset of 6,335 labeled news articles, the model employs TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text into numerical features, then applies Linear Support Vector Classification for prediction.

## Features

- **High Accuracy**: Achieves 94.5% accuracy on test data
- **TF-IDF Vectorization**: Converts text to numerical features based on word importance
- **Linear SVC Classification**: Uses one of the best algorithms for text classification
- **Efficient Processing**: Handles large datasets with optimized preprocessing
- **Binary Classification**: Distinguishes between REAL (0) and FAKE (1) news articles

## Dataset Structure

The model works with news articles containing:
- **ID**: Unique identifier for each article
- **Title**: News article headline
- **Text**: Full article content
- **Label**: Binary classification (REAL/FAKE)

Sample data format:
```
         id                                              title  
0      8476                       You Can Smell Hillary's Fear   
1     10294  Watch The Exact Moment Paul Ryan Committed Pol...   
2      3608        Kerry to go to Paris in gesture of sympathy   
3     10142  Bernie supporters on Twitter erupt in anger ag...   
```

## Implementation Details

### Data Preprocessing
```python
# Convert categorical labels to binary format
data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
data = data.drop("label", axis=1)

# Split features and target
X, y = data["text"], data["fake"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### TF-IDF Vectorization
```python
# TF-IDF: Term Frequency - Inverse Document Frequency
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train) 
X_test_vectorized = vectorizer.transform(X_test)
```

### Model Training and Evaluation
```python
# Linear SVC - considered one of the best text classification algorithms
clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)

# Evaluate model performance
accuracy = clf.score(X_test_vectorized, y_test)
print(f"Accuracy: {accuracy:.4f}")  # Output: 0.9455 (94.55%)
```

### Prediction Example
```python
# Test on individual article
article_text = X_test.iloc[10]
vectorized_text = vectorizer.transform([article_text])
prediction = clf.predict(vectorized_text)  # Returns: array([1]) for FAKE
```

## TF-IDF Explained

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a numerical statistic that reflects how important a word is to a document in a collection of documents.

### Components:
- **TF (Term Frequency)**: Number of times a term appears in a document
- **IDF (Inverse Document Frequency)**: Logarithm of total documents divided by documents containing the term
- **TF-IDF Score**: TF × IDF

### Purpose:
TF-IDF helps identify the most relevant and distinctive words per document by:
- Increasing weight for words that appear frequently in a document (TF)
- Decreasing weight for words that appear frequently across all documents (IDF)
- Filtering common words that don't provide discriminative information

## Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook (optional, for running the .ipynb file)

### Dependencies
```bash
pip install numpy pandas scikit-learn
```

For Jupyter notebook:
```bash
pip install jupyter
```

## Usage

### Running the Notebook
```bash
jupyter notebook "Fake News Detection.ipynb"
```

### Using the Model in Python
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv("fake_or_real_news.csv")

# Preprocess data
data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
X, y = data["text"], data["fake"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train model
clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)

# Make predictions
def predict_news(article_text):
    vectorized = vectorizer.transform([article_text])
    prediction = clf.predict(vectorized)[0]
    return "FAKE" if prediction == 1 else "REAL"
```

## Model Performance

- **Algorithm**: Linear Support Vector Classification (LinearSVC)
- **Feature Engineering**: TF-IDF Vectorization
- **Training Set**: 80% of 6,335 articles
- **Test Set**: 20% of 6,335 articles
- **Accuracy**: 94.55%
- **Preprocessing**: English stop words removal, max document frequency of 0.7

## Key Features of Implementation

### Text Preprocessing
- Removes English stop words to focus on meaningful terms
- Applies maximum document frequency threshold (0.7) to filter overly common words
- Converts text to numerical vectors suitable for machine learning

### Classification Algorithm
Linear SVC is particularly effective for text classification because:
- Handles high-dimensional sparse data efficiently
- Works well with TF-IDF features
- Provides good generalization for text classification tasks
- Computationally efficient for large datasets

## Project Structure

```
Fake News Detection/
├── README.md
├── Fake News Detection.ipynb    # Main Jupyter notebook
└── fake_or_real_news.csv       # Dataset (not included)
```

## Future Enhancements

- **Deep Learning Models**: Implement LSTM or BERT-based models for improved accuracy
- **Feature Engineering**: Add metadata features (publication date, author, source)
- **Real-time Classification**: Create web interface for live news article classification
- **Multi-class Classification**: Extend to classify different types of misinformation
- **Explainability**: Add model interpretation to understand decision factors

## Technical Notes

- The model uses binary classification (0 for REAL, 1 for FAKE)
- TF-IDF parameters can be tuned for different datasets
- LinearSVC provides fast training and prediction times
- The implementation handles large text datasets efficiently

## License

This project is open-source and available under the MIT License.
