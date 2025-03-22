# Sentiment-Analysis-with-IMDB-Reviews


## Overview
This project implements a sentiment analysis model using IMDB movie reviews. It processes textual data, extracts relevant features, and applies a machine learning model to classify sentiments as either positive or negative.

## Dataset
The dataset used is the **IMDB Dataset**, containing movie reviews labeled with sentiments:
- **Positive (1)**
- **Negative (0)**

## Workflow
1. **Data Preprocessing**
   - Converts text to lowercase
   - Removes HTML tags, URLs, and special characters
   - Tokenizes text and removes stopwords
   - Applies stemming using **PorterStemmer**

2. **Feature Extraction**
   - Uses **TF-IDF Vectorization** to convert text into numerical features.

3. **Model Training**
   - Splits data into training and test sets.
   - Trains a machine learning model (possibly Logistic Regression, Naive Bayes, or another classifier).
   
4. **Evaluation**
   - Measures accuracy, precision, recall, and F1-score.
   - Visualizes results using word clouds and histograms.

## Requirements
Ensure you have the following installed:
```bash
pip install pandas numpy nltk matplotlib seaborn scikit-learn wordcloud plotly
```

## Usage
To run the model:
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('IMDB Dataset.csv')

# Preprocess and vectorize text
df['processed_review'] = df['review'].apply(data_processing)
vect = TfidfVectorizer()
X = vect.fit_transform(df['processed_review'])

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, df['sentiment'], test_size=0.2, random_state=42)
```

## Results
- Visualizations of word frequency distributions
- Performance metrics for classification accuracy
- Word cloud representations of most common words in positive and negative reviews

## Conclusion
This model effectively classifies sentiments from IMDB reviews. Further improvements can be made using deep learning techniques like LSTMs or transformers.

