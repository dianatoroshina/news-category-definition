import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score


data = pd.read_hdf('data.h5', 'data')
print(data.head())

le = LabelEncoder()
data['category'] = le.fit_transform(data['category'])
print(data.head())

vectorizer = CountVectorizer()

columns = ['headline', 'short_description', 'authors']
data['combined_text'] = data[columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
x_train, x_test, y_train, y_test = train_test_split(data['combined_text'], data['category'], test_size=0.2)
print(x_train.head())

x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

model = LogisticRegression(max_iter=10000)
model.fit(x_train_vectorized, y_train)
y_pred = model.predict(x_test_vectorized)
print('Logistic Regression:', round(cohen_kappa_score(y_pred, y_test, weights='quadratic'), 3))