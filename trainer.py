import logging

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

n_iter = 50
scoring = "f1_macro"
n_folds = 4  # Use 4 stratified folds for 60/20/20 split (80% cv fold, 20% held out)
verbose = 1  # Log some progress
n_jobs = -1  # Use all cores

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Trainer(object):
    """Base class for running an sklearn pipeline + cross validator"""

    def __init__(self, search_cv):
        self.model = search_cv
        self.model.cv = n_folds
        self.model.verbose = verbose
        self.model.n_jobs = n_jobs
        self.model.n_iter = n_iter
        self.model.scoring = scoring
        self.model_family = self.__class__.__name__.replace("Trainer", "")
        self.score = None

    def train(self, x_text, y_text):
        logging.info("Performing Randomized Search over %s Parameters" % self.model_family)
        self.model.fit(x_text, y_text)

        logging.info("Best Score: %f" % self.model.best_score_)
        logging.info("Best Hyperparameters: %s" % self.model.best_params_)

        self.score = self.model.best_score_


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
                    "classifier__alpha": stats.expon(scale=1.0),
                    "classifier__fit_prior": [True, False],
                },
        )

        super(MultinomialNBTrainer, self).__init__(search_cv)


class BernoulliNBTrainer(Trainer):
    def __init__(self):
        pipeline = Pipeline([
            ("vectorizer", CountVectorizer(stop_words="english")),
            ("classifier", BernoulliNB()),
        ])

        search_cv = RandomizedSearchCV(
                pipeline,
                param_distributions={
                    "classifier__alpha": stats.expon(scale=1.0),
                    "classifier__binarize": stats.uniform(0, 5),
                    "classifier__fit_prior": [True, False],
                },
        )

        super(BernoulliNBTrainer, self).__init__(search_cv)


class LogisticRegressionTrainer(Trainer):
    def __init__(self):
        pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(stop_words="english")),
            ("classifier", SGDClassifier(loss="log", penalty="elasticnet")),
        ])

        search_cv = RandomizedSearchCV(
                pipeline,
                param_distributions={
                    "vectorizer__binary": [True, False],
                    "vectorizer__norm": ['l1', 'l2', None],
                    "vectorizer__use_idf": [True, False],
                    "vectorizer__sublinear_tf": [True, False],
                    "classifier__l1_ratio": stats.uniform(0, 1),
                    "classifier__alpha": stats.expon(scale=0.1),
                },
        )

        super(LogisticRegressionTrainer, self).__init__(search_cv)


class KNNTrainer(Trainer):
    def __init__(self):
        pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(stop_words="english")),
            ("classifier", KNeighborsClassifier(loss="log", penalty="elasticnet")),
        ])

        search_cv = RandomizedSearchCV(
                pipeline,
                param_distributions={
                    "vectorizer__binary": [True, False],
                    "vectorizer__norm": ['l1', 'l2', None],
                    "vectorizer__use_idf": [True, False],
                    "vectorizer__sublinear_tf": [True, False],
                    "classifier__n_neighbors": [i for i in range(2, 10)],
                    "classifier__weights": ["uniform", "distance"],
                },
        )

        super(KNNTrainer, self).__init__(search_cv)


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
                    "classifier__n_estimators": stats.randint(low=5, high=30),
                    "classifier__criterion": ["gini", "entropy"],
                    "classifier__min_samples_leaf": stats.randint(low=1, high=5),
                    "classifier__class_weight": [None, "balanced"],
                },
        )

        super(RandomForestTrainer, self).__init__(search_cv)
