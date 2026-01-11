# TUGASRISET_NLP
# =========================================
# NLP SENTIMENT ANALYSIS E-COMMERCE
# Author : Rangga Timotius
# =========================================

# ========= 1. INSTALL & IMPORT =========
!pip install Sastrawi wordcloud

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

from wordcloud import WordCloud

# ========= 2. LOAD DATA =========
# Ganti path sesuai dataset kamu
df = pd.read_csv('/content/dataset.csv')

# Jika nama kolom berbeda, sesuaikan di sini
df = df.rename(columns={
    'content': 'review',
    'score': 'rating'
})

df = df[['review', 'rating']].dropna()
print("Jumlah data awal:", df.shape)

# ========= 3. LABEL SENTIMENT =========
def label_sentiment(r):
    if r >= 4:
        return 'positive'
    elif r <= 2:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['rating'].apply(label_sentiment)
df = df[df['sentiment'] != 'neutral']
print("Setelah hapus netral:", df.shape)

# ========= 4. EDA AWAL =========
plt.figure(figsize=(5,4))
sns.countplot(x='rating', data=df)
plt.title('Distribusi Rating')
plt.show()

plt.figure(figsize=(5,4))
sns.countplot(x='sentiment', data=df)
plt.title('Distribusi Sentimen')
plt.show()

df['text_length'] = df['review'].apply(lambda x: len(str(x).split()))
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title('Distribusi Panjang Review')
plt.show()

# ========= 5. SAMPLING DATA (OPTIMASI) =========
df_sample = df.sample(n=1000, random_state=42).copy()
print("Data sampling:", df_sample.shape)

# ========= 6. PREPROCESSING =========
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

print("Mulai preprocessing...")
df_sample['final_text'] = df_sample['review'].apply(preprocess)
print("Preprocessing selesai")

# ========= 7. HASIL PREPROCESSING =========
df_sample[['review','final_text','sentiment']].head()

# ========= 8. WORDCLOUD =========
wc = WordCloud(width=800, height=400, background_color='white')
wc.generate(' '.join(df_sample['final_text']))

plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud Review E-Commerce')
plt.show()

# ========= 9. TF-IDF =========
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X = tfidf.fit_transform(df_sample['final_text'])
y = df_sample['sentiment']

# ========= 10. SPLIT DATA =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========= 11. NAIVE BAYES =========
nb = MultinomialNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)

print("=== Naive Bayes ===")
print(classification_report(y_test, pred_nb))

# ========= 12. SVM =========
svm = LinearSVC()
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

print("=== SVM ===")
print(classification_report(y_test, pred_svm))

# ========= 13. CONFUSION MATRIX =========
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test, pred_nb), annot=True, fmt='d')
plt.title('Naive Bayes')

plt.subplot(1,2,2)
sns.heatmap(confusion_matrix(y_test, pred_svm), annot=True, fmt='d')
plt.title('SVM')

plt.show()

# ========= 14. CROSS VALIDATION =========
cv_nb = cross_val_score(nb, X, y, cv=5, scoring='f1_macro')
cv_svm = cross_val_score(svm, X, y, cv=5, scoring='f1_macro')

print("NB CV F1:", cv_nb.mean())
print("SVM CV F1:", cv_svm.mean())

# ========= 15. HASIL AKHIR KLASIFIKASI =========
result = df_sample[['review','rating','sentiment']].copy()
result['predicted_sentiment'] = svm.predict(X)
result.head()
