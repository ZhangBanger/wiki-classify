# Settings for pipeline and hyper-parameter search
import logging

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer

n_folds = 5  # Use 5 stratified folds for 80/20 CV
verbose = 1  # Log some progress
n_jobs = -1  # Use all cores

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Trainer(object):
    """Base class for running an sklearn pipeline + cross validator"""

    def __init__(self, search_cv):
        self.model = search_cv
        self.model_family = self.__class__.__name__.replace("Trainer", "")
        self.score = None

    def train(self, x_text, y_text):
        logging.info("Performing Randomized Search over %s Models" % self.model_family)
        self.model.fit(x_text, y_text)
        logging.info("Best Model Found: %s" % repr(self.model.best_estimator_))
        logging.info("Best Model Score: %f" % self.model.best_score_)
        logging.info("Best Model Hyper-parameters: %s" % self.model.best_params_)

        self.score = self.model.best_score_

        return self


class MultinomialNBTrainer(Trainer):
    def __init__(self):
        pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(stop_words="english")),
            ("classifier", MultinomialNB()),
        ])
        search_cv = RandomizedSearchCV(
                pipeline,
                param_distributions={
                    "vectorizer__binary": [True, False],
                    "vectorizer__norm": ['l1', 'l2', None],
                    "vectorizer__use_idf": [True, False],
                    "vectorizer__sublinear_tf": [True, False],
                    "classifier__alpha": stats.expon(0.5),
                    "classifier__fit_prior": [True, False],
                },
                cv=n_folds,
                n_iter=40,  # iterations of randomized hyper-parameter search
                verbose=verbose,
                n_jobs=n_jobs,
        )
        super(MultinomialNBTrainer, self).__init__(search_cv)


class BernoulliNBTrainer(Trainer):
    def __init__(self):
        pipeline = Pipeline([
            ("vectorizer", CountVectorizer(stop_words="english")),
            ("binarizer", Binarizer()),
            ("classifier", BernoulliNB()),
        ])

        search_cv = RandomizedSearchCV(
                pipeline,
                param_distributions={
                    "binarizer__threshold": [1, 2, 3, 5],
                    "classifier__alpha": stats.expon(0.5),
                    "classifier__fit_prior": [True, False],
                },
                cv=n_folds,
                n_iter=30,  # iterations of randomized hyper-parameter search
                verbose=verbose,
                n_jobs=n_jobs,
        )
        super(BernoulliNBTrainer, self).__init__(search_cv)


class LinearModelTrainer(Trainer):
    def __init__(self):
        pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(stop_words="english")),
            ("classifier", SGDClassifier()),
        ])

        search_cv = RandomizedSearchCV(
                pipeline,
                param_distributions={
                    "vectorizer__binary": [True, False],
                    "vectorizer__norm": ['l1', 'l2', None],
                    "vectorizer__use_idf": [True, False],
                    "vectorizer__sublinear_tf": [True, False],
                    "classifier__loss": ['hinge', 'log', 'perceptron'],
                    "classifier__penalty": [None, 'l1', 'l2', 'elasticnet'],
                    "classifier__l1_ratio": [0.1, 0.15, 0.2],
                    "classifier__alpha": [0.0001, 0.001, 0.01],
                },
                cv=n_folds,
                n_iter=50,  # iterations of randomized hyper-parameter search
                verbose=verbose,
                n_jobs=n_jobs,
        )
        super(LinearModelTrainer, self).__init__(search_cv)


class RandomForestTrainer(Trainer):
    def __init__(self):
        pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(stop_words="english")),
            ("classifier", RandomForestClassifier()),
        ])

        search_cv = RandomizedSearchCV(
                pipeline,
                param_distributions={
                    "vectorizer__binary": [True, False],
                    "vectorizer__norm": ['l1', 'l2', None],
                    "vectorizer__use_idf": [True, False],
                    "vectorizer__sublinear_tf": [True, False],
                    "classifier__n_estimators": [5, 10, 15, 20],
                    "classifier__criterion": ["gini", "entropy"],
                    "classifier__min_samples_leaf": [1, 2, 3],
                    "classifier__class_weight": [None, "balanced"],
                },
                cv=n_folds,
                n_iter=50,  # iterations of randomized hyper-parameter search
                verbose=verbose,
                n_jobs=n_jobs,
        )
        super(RandomForestTrainer, self).__init__(search_cv)
