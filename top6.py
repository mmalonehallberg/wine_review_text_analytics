#################################
### Wine review text analysis ###
### Meg Malone, Sandy Preiss ####
### 12oct2019 ###################
#################################

conda install -c conda-forge spacy
conda install -c conda-forge nltk
conda install -c conda-forge scikit-learn
conda install -c conda-forge pandas
# conda install -c conda-forge xgboost

import nltk
nltk.download('stopwords')
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import csv
import pandas as pd
import unicodedata
import spacy
from collections import Counter
import os

#pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.1/en_core_web_sm-2.2.1.tar.gz --no-deps

!python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm") 


os.chdir('C:\\Users\\Sandy\\Documents\\MSA\\Text Analysis')



# Import dataset (already have isolated the two columns for this analysis)
wine = pd.read_csv('winemag_filt.csv')

# Filter for 15 most popular wine varieties in the US
top15 = ['Pinot Noir','Chardonnay','Cabernet Sauvignon',
         'Red Blend','Riesling','Sauvignon Blanc','Syrah',
         'Rose','Merlot','Zinfandel','Malbec', 'White Blend',
        'Pinot Gris','Pinot Grigio','Shiraz','Moscato', 'Muscat']

# Filter for top 6 non-blend categories in dataset (37% of data)
top6 = ['Pinot Noir','Chardonnay','Cabernet Sauvignon','Riesling','Sauvignon Blanc','Syrah']

wine6 = wine[wine.variety.isin(top6)]

# Take a random sample of 10000 observations from that filtered set
wine_samp6 = wine6.sample(n=10000, random_state=27)

# Format columns as lists
corpus = wine_samp6['description'].values.tolist()
labels = wine_samp6['variety'].values.tolist()

corpus2 = wine6['description'].values.tolist()
labels2 = wine6['variety'].values.tolist()

# Accented char function
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text



# Special char function
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text



# Lemmatization function (version of stemming that maintains English spellings)
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text



# Tokenizer function
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text



# All the functions together now!
def normalize_corpus(corpus,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True, remove_digits=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus


# Applying to the data
norm_corpus = normalize_corpus(corpus)

norm_corpus2 = normalize_corpus(corpus2)


# Making training and test sets
train_corpus, test_corpus, train_label_names, test_label_names = train_test_split(norm_corpus2, labels2, test_size=0.3, random_state=27)


# TF-IDF 
tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0)
tv_train_features = tv.fit_transform(train_corpus)
tv_test_features = tv.transform(test_corpus)



### MODELING ###

# Naive Bayes
mnb = MultinomialNB(alpha=1)
mnb.fit(tv_train_features, train_label_names)
mnb_tfidf_cv_scores = cross_val_score(mnb, tv_train_features, train_label_names, cv=5)
mnb_tfidf_cv_mean = np.mean(mnb_tfidf_cv_scores)
print('Naive Bayes')
print('CV Accuracy, 5-fold:', mnb_tfidf_cv_scores)
print('Mean CV Accuracy:', mnb_tfidf_cv_mean)

# Logistic
lr = LogisticRegression(penalty = 'l2', C = 1, random_state = 27)
lr.fit(tv_train_features, train_label_names)
lr_tfidf_cv_scores = cross_val_score(lr, tv_train_features, train_label_names, cv=5)
lr_tfidf_cv_mean = np.mean(lr_tfidf_cv_scores)
print('Logistic Regression')
print('CV Accuracy, 5-fold:', lr_tfidf_cv_scores)
print('Mean CV Accuracy:', lr_tfidf_cv_mean)

# SVM
svm = LinearSVC(penalty = 'l2', C = 1, random_state = 27)
svm.fit(tv_train_features, train_label_names)
svm_tfidf_cv_scores = cross_val_score(svm, tv_train_features, train_label_names, cv=5)
svm_tfidf_cv_mean = np.mean(svm_tfidf_cv_scores)
print('Support Vector Machine')
print('CV Accuracy, 5-fold:', svm_tfidf_cv_scores)
print('Mean CV Accuracy:', svm_tfidf_cv_mean)

# SVM with SGD
sgd = SGDClassifier(loss = 'hinge', penalty = 'l2', max_iter = 20, random_state = 27)
sgd.fit(tv_train_features, train_label_names)
sgd_tfidf_cv_scores = cross_val_score(sgd, tv_train_features, train_label_names, cv=5)
sgd_tfidf_cv_mean = np.mean(sgd_tfidf_cv_scores)
print('Support Vector Machine with Stochastic Gradient Descent')
print('CV Accuracy, 5-fold:', sgd_tfidf_cv_scores)
print('Mean CV Accuracy:', sgd_tfidf_cv_mean)

# Random Forest
rfc = RandomForestClassifier(n_estimators = 100, random_state = 27)
rfc.fit(tv_train_features, train_label_names)
rfc_tfidf_cv_scores = cross_val_score(rfc, tv_train_features, train_label_names, cv=5)
rfc_tfidf_cv_mean = np.mean(rfc_tfidf_cv_scores)
print('Random Forest Classifier')
print('CV Accuracy, 5-fold:', rfc_tfidf_cv_scores)
print('Mean CV Accuracy:', rfc_tfidf_cv_mean)

# Gradient Boosting
gbc = GradientBoostingClassifier(n_estimators = 100, random_state = 27)
gbc.fit(tv_train_features, train_label_names)
gbc_tfidf_cv_scores = cross_val_score(gbc, tv_train_features, train_label_names, cv=5)
gbc_tfidf_cv_mean = np.mean(gbc_tfidf_cv_scores)
print('Gradient Boosting Classifier')
print('CV Accuracy, 5-fold:', gbc_tfidf_cv_scores)
print('Mean CV Accuracy:', gbc_tfidf_cv_mean)


### Parameter Tuning ###

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# Logistic Regression
lr_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                        ('lr', LogisticRegression(penalty = 'l2', max_iter = 20, random_state = 27))
                        ])
    
param_grid = {'tfidf__ngram_range': [(1, 1), (1, 2)],
              'lr__C': [1, 5, 10]                       
              }

gs_lr = GridSearchCV(lr_pipeline, param_grid, cv=5, verbose=2)
gs_lr = gs_lr.fit(train_corpus, train_label_names)

cv_results = gs_lr.cv_results_
lr_tuning_results = pd.DataFrame({'rank' : cv_results['rank_test_score'],
                                  'params' : cv_results['params'],
                                  'cv score (mean)' : cv_results['mean_test_score'],
                                  'cv score (std)' : cv_results['std_test_score']
                                  })

lr_tuning_results = lr_tuning_results.sort_values(by=['rank'], ascending=True)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 100)
lr_tuning_results


# SVM
svm_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                        ('svm', LinearSVC(random_state = 27))
                        ])
    
param_grid = {'tfidf__ngram_range': [(1, 1), (1, 2)],
              'svm__C': [0.01, 0.1, 1, 5]                       
              }

gs_svm = GridSearchCV(svm_pipeline, param_grid, cv=5, verbose=2)
gs_svm = gs_svm.fit(train_corpus, train_label_names)

cv_results = gs_svm.cv_results_
svm_tuning_results = pd.DataFrame({'rank' : cv_results['rank_test_score'],
                                  'params' : cv_results['params'],
                                  'cv score (mean)' : cv_results['mean_test_score'],
                                  'cv score (std)' : cv_results['std_test_score']
                                  })

svm_tuning_results = svm_tuning_results.sort_values(by=['rank'], ascending=True)
svm_tuning_results



### Final Model Testing ###


tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1,2))
tv_train_features = tv.fit_transform(train_corpus)
tv_test_features = tv.transform(test_corpus)


# Final Logistic Model
lr = LogisticRegression(penalty = 'l2', C = 10, random_state = 27)
lr.fit(tv_train_features, train_label_names)
lr_tfidf_cv_scores = cross_val_score(lr, tv_train_features, train_label_names, cv=5)
lr_tfidf_cv_mean = np.mean(lr_tfidf_cv_scores)
print('Logistic Regression')
print('CV Accuracy, 5-fold:', lr_tfidf_cv_scores)
print('Mean CV Accuracy:', lr_tfidf_cv_mean)

lr_tfidf_test_score = lr.score(tv_test_features, test_label_names)
print('Test Accuracy:', lr_tfidf_test_score)


# Model Evaluation

from sklearn import metrics

lr_predictions = lr.predict(tv_test_features)
unique_classes = list(set(test_label_names))

print('Accuracy:', np.round(
                    metrics.accuracy_score(test_label_names, 
                                           lr_predictions),
                    4))
print('Precision:', np.round(
                    metrics.precision_score(test_label_names, 
                                           lr_predictions,
                                           average='weighted'),
                    4))
print('Recall:', np.round(
                    metrics.recall_score(test_label_names, 
                                           lr_predictions,
                                           average='weighted'),
                    4))
print('F1 Score:', np.round(
                    metrics.f1_score(test_label_names, 
                                           lr_predictions,
                                           average='weighted'),
                    4))


report = metrics.classification_report(y_true=test_label_names, 
                                       y_pred=lr_predictions, 
                                       labels=unique_classes) 
print(report)
    

cm = metrics.confusion_matrix(y_true=test_label_names, y_pred=lr_predictions, 
                              labels=unique_classes)
print(cm)

cm_frame = pd.DataFrame(data=cm, 
                        columns=unique_classes, 
                        index=unique_classes )
print(cm_frame) 
    
preds = pd.DataFrame({'Review' : test_corpus,
                      'Actual' : test_label_names,
                      'Predicted' : lr_predictions
                      })

print(preds)

