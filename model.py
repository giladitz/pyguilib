#Importing dependencies
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import string

#Importing NLTK dependencies
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from nltk import pos_tag

#Importing vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#importing machine learning functions
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
import sklearn.manifold
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import scikitplot as skpl



# Preprocessing functions
def clean(text):
    text = re.sub(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        '', text, flags=re.MULTILINE)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"[^\w\s\d]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return (text)


# Emoticon removal
def demoji(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))


# Stopwords removal
def stop(text):
    return (" ".join([word for word in str(text).split() if word not in stopwords]))


# Tokenization of the sentences
def token(text):
    return word_tokenize(df['Tweets'])


# Lemmetising
le = WordNetLemmatizer()
stemmer = PorterStemmer()


# defining parts of speech of morphy tags
def POS(text):
    tags = {'NN': 'n', 'JJ': 'a',
            'VB': 'v', 'RB': 'r'}
    try:
        return tags[text[:2]]
    except:
        return 'n'

    # Lemmatizing


def lemmatize(text):
    # Text input is string, returns lowercased strings.
    return [le.lemmatize(word.lower(), pos=POS(tag))
            for word, tag in pos_tag(word_tokenize(text))]


# stemming
def stem(text):
    return " ".join([stemmer.stem(word) for word in text])


# Applying functions
df['Tweets'] = df['Tweets'].apply(lambda text: clean(text))
df['Tweets'] = df['Tweets'].apply(lambda text: demoji(text))
df['Tweets'] = df['Tweets'].apply(lambda text: stop(text))
df['Tweets'] = df['Tweets'].apply(lambda text: lemmatize(text))
df['Tweets'] = df['Tweets'].apply(lambda text: stem(text))


#A function to create various N-gram combinations
def tfidf(i,j):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,stop_words='english',ngram_range=(i,j))
    return(tfidf.fit_transform(df['Tweets']).toarray())

#calling the function of tfidf for various combinations of N-gram.
features=tfidf(1,1)

#labelling the sentiments
la=LabelEncoder()
df['Label'] = la.fit_transform(df['Sentiment'])

#printing the dataset
df.head()

#features and labels
x=features
y=df['Label']

#Train test split
x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)

#Classification using Multinomial Naive Bayes
classifier = MultinomialNB(fit_prior=True,alpha=0.1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

#Printing the metrics
print("Classification Report: \n", classification_report(y_test,y_pred))
print("Accuracy: ", (accuracy_score(y_test, y_pred))*100)
print("F1 score: ",f1_score(y_test, y_pred, average='macro'))

#Support Vector machine classifier
clf=SVC(C=1,gamma=1,kernel='linear',probability=True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

#printing metrics
print("Classification Report: \n", classification_report(y_test,y_pred))
print("Accuracy: ", (accuracy_score(y_test, y_pred))*100)
print("F1 score: ",f1_score(y_test, y_pred, average='macro'))

#function to plot ROC for classifiers
def ROC(model):
    ypred=model.predict_proba(x_test)
    return(skpl.metrics.plot_roc(y_test,ypred))

#Plotting the classifer's ROC
ROC(classifier)
ROC(clf)
