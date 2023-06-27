import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')


data = pd.read_json('News_Category_Dataset_v3.json', lines=True)
print(data.head())

print(data.isna().sum())
print(data.duplicated())

data.drop_duplicates(inplace=True)
print(data.duplicated().sum())

print(data['short_description'])

def clean_text(text):
    text = text.lower()
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_text = ' '.join(words)
    
    return cleaned_text

data['short_description'] = data['short_description'].apply(clean_text)
print(data['short_description'])

data.to_hdf('data.h5', 'data')