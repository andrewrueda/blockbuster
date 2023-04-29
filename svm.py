import pandas
import sklearn
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import new_dataset
import numpy


from sklearn.feature_extraction.text import CountVectorizer


from typing import (
    Iterable,
    Any,
    Sequence,
    Generator,
)

import random
random.seed(100)

vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 10000)

runtimes =[]
character_matches = []
metadata = []
y = []


for movie in new_dataset.load_instances(0, len(new_dataset.data)):
    metadata.append(" ".join(movie.studio) + movie.release_season[0] + " ".join(movie.genre) + " ".join(movie.synopsis_ngrams))
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

#svm_model = svm.SVC(C=2, kernel='rbf', degree=3, gamma='auto')
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)

pred_svm = svm_model.predict(X_val)
print(sklearn.metrics.classification_report(y_val, pred_svm))