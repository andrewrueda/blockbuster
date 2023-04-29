import pandas
import sklearn
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import new_dataset
import numpy
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


from sklearn.feature_extraction.text import CountVectorizer


from typing import (
    Iterable,
    Any,
    Sequence,
    Generator,
)

import random
random.seed(1200)

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
X = bag

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=170)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=(1/9), random_state=170)


# grid = {'bootstrap': [True, False],
#  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#  'min_samples_leaf': [1, 2, 4],
#  'min_samples_split': [2, 5, 10],
#  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train, y_train)

# rfc = RandomForestRegressor()
# grid_search = GridSearchCV(estimator = rfc, param_grid = grid, cv = 3, n_jobs=-1, verbose=2)
#
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)


##Val and Test Evaluation

# pred_rfc = rfc.predict(X_val)
# print(sklearn.metrics.classification_report(y_val, pred_rfc))

pred_rfc = rfc.predict(X_test)
print(sklearn.metrics.classification_report(y_test, pred_rfc))