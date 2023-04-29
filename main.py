import pandas
import numpy
import sklearn
from sklearn import model_selection
import csv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

from numpy import mean
from numpy import std
import random
import new_dataset
from collections import Counter

random.seed(10)

vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 10000)

# with open("logistic_regression_input.csv", "w", encoding="utf-8") as file:
#     write_in = csv.writer(file)
#     write_in.writerow(["title", "release_season", "genre", "synopsis", "runtime", "revenue"])
#     for movie in preparation.load_instances(0, len(preparation.data)):
#         write_in.writerow([movie.title, movie.release_season, movie.genre, movie.synopsis_ngrams, movie.runtime, movie.revenue])
# rd = pandas.read_csv("logistic_regression_input.csv")


runtimes =[]
character_matches = []
metadata = []
y = []

for movie in new_dataset.load_instances(0, len(new_dataset.data)):
    metadata.append(" ".join(movie.studio) + movie.release_season[0] + " ".join(movie.genre) +  " ".join(movie.synopsis_ngrams))
    runtimes.append([movie.runtime])
    character_matches.append([movie.character_match])
    y.append(movie.revenue)

movie_data = numpy.array(metadata)
bag = vectorizer.fit_transform(movie_data)
bag = bag.toarray()
extras = numpy.append(runtimes, character_matches, axis=1)
X = numpy.append(bag, extras, axis=1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=170)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=(1/9), random_state=170)
lr = LogisticRegression(C=.1, max_iter=20000, class_weight={'under': .4, 'over': .6}).fit(X_train, y_train)

##Val and Test Evaluation

# y_predict = lr.predict(X_val)
# print(sklearn.metrics.classification_report(y_val, y_predict))

y_predict = lr.predict(X_test)
print(sklearn.metrics.classification_report(y_test, y_predict))