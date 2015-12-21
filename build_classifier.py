import argparse
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from wiki import get_and_save_text

categories = [
    "Rare_diseases",
    "Infectious_diseases",
    "Cancer",
    "Congenital_disorders",
    "Organs_(anatomy)",
    "Machine_learning_algorithms",
    "Medical_devices",
]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="dir to save or load existing X.txt, y.txt", default=os.getcwd())
parser.add_argument("-m", "--model", help="path to save model and vectorizer", default=os.getcwd())
args = parser.parse_args()

print("Collecting data for all categories")

# Initialize and collect data
X_text, y_text = [], []

if os.path.isfile("%s/X.txt" % args.data) and os.path.isfile("%s/y.txt" % args.data):
    print("Found data in %s: " % args.data)
    with open("%s/X.txt" % args.data) as X_file, open("%s/y.txt" % args.data) as y_file:
        X_text = X_file.readlines()
        y_text = y_file.readlines()
else:
    X_text, y_text = get_and_save_text(categories, args.data)

print("Collected %i examples across %i categories" % (len(X_text), len(categories)))

# Vectorize inputs and labels
X = CountVectorizer(stop_words='english').fit_transform(X_text)
y = LabelEncoder().fit_transform(y_text)

# BEGIN GRID SEARCH
n_folds = 5

# Grid Search over Naive Bayes Models
print("Performing Grid Search over Naive Bayes Models with %s stratified folds" % n_folds)
grid_search = GridSearchCV(MultinomialNB(), param_grid={"alpha": [0.0, 1.0, 2.0]}, cv=n_folds, n_jobs=-1)
grid_search.fit(X, y)
print("Best Model Found: %s" % grid_search.best_estimator_.__repr__())
print("Best Model Score: %f" % grid_search.best_score_)
print("Best Model Hyper-parameters: %s" % grid_search.best_params_)

# Grid Search over Linear Models (Logistic Regression, Perceptron, SVM)
print("Performing Grid Search over Linear Models with %s stratified folds" % n_folds)
grid_search = GridSearchCV(
        SGDClassifier(),
        param_grid={
            "loss": ['hinge', 'log', 'perceptron'],
            "penalty": [None, 'l1', 'l2', 'elasticnet'],
            "l1_ratio": [0.1, 0.15, 0.2],
            "alpha": [0.0001, 0.001, 0.01],
        },
        cv=n_folds,
        n_jobs=-1,
)
grid_search.fit(X, y)
print("Best Model Found: %s" % grid_search.best_estimator_.__repr__())
print("Best Model Score: %f" % grid_search.best_score_)
print("Best Model Hyper-parameters: %s" % grid_search.best_params_)

# Grid Search over Random Forests (could add other ensemble models later)
print("Performing Grid Search over Random Forest Models with %s stratified folds" % n_folds)
grid_search = GridSearchCV(
        RandomForestClassifier(n_jobs=-1),
        param_grid={
            "n_estimators": [5, 10, 15, 20],
            "criterion": ["gini", "entropy"],
            "min_samples_leaf": [1, 2, 3],
            "class_weight": [None, "balanced"],
        },
        cv=n_folds,
        n_jobs=-1,
)
grid_search.fit(X, y)
print("Best Model Found: %s" % grid_search.best_estimator_.__repr__())
print("Best Model Score: %f" % grid_search.best_score_)
print("Best Model Hyper-parameters: %s" % grid_search.best_params_)
