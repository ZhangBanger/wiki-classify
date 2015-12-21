import argparse
import os

from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
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
    print("Found dataset in %s: " % args.data)
    with open("%s/X.txt" % args.data) as X_file, open("%s/y.txt" % args.data) as y_file:
        X_text = X_file.readlines()
        y_text = y_file.readlines()
else:
    X_text, y_text = get_and_save_text(categories, args.data)

print("Collected %i examples across %i categories" % (len(X_text), len(categories)))

# Vectorize inputs and labels
X = CountVectorizer(stop_words='english').fit_transform(X_text)
y = LabelEncoder().fit_transform(y_text)

# Training-test split
n_folds = 5
print("Training NB")
print("Performing stratified k fold with %i folds" % n_folds)
k_fold = StratifiedKFold(y, n_folds=n_folds)
for train_index, test_index in k_fold:
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # Train classifier
    # Default LaPlace smoothing
    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(X_train, y_train)

    # Evaluate classifier
    predicted = classifier.predict(X_test)
    print(metrics.classification_report(y_test, predicted, target_names=categories))

# Training-test split
n_folds = 5
print("Training Linear Classifier")
print("Performing stratified k fold with %i folds" % n_folds)
k_fold = StratifiedKFold(y, n_folds=n_folds)
for train_index, test_index in k_fold:
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # Train classifier
    # Default LaPlace smoothing
    classifier = SGDClassifier(loss='log')
    classifier.fit(X_train, y_train)

    # Evaluate classifier
    predicted = classifier.predict(X_test)
    print(metrics.classification_report(y_test, predicted, target_names=categories))
