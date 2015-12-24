import argparse
import logging
import os
import pickle

from sklearn.preprocessing import MultiLabelBinarizer

from trainer import LogisticRegressionTrainer, RandomForestTrainer, \
    KNNTrainer, MultinomialNBTrainer, BernoulliNBTrainer
from wiki_utils import get_and_save_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

categories = [category.rstrip() for category in open("%s/categories.txt" % os.getcwd())]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="dir to save or load existing X.txt, y.txt", default=os.getcwd())
parser.add_argument("-m", "--model", help="path to save model and binarizer", default=os.getcwd())
args = parser.parse_args()

logging.info("Collecting data for all categories")

# Initialize and collect data
X_text, y = [], []
test_size = 0.2

# Load X_train.txt and y_train.txt or fetch train and test sets
if os.path.isfile("%s/X_train.txt" % args.data) and os.path.isfile("%s/y_train.txt" % args.data):
    logging.info("Found data in %s: " % args.data)
    with open("%s/X_train.txt" % args.data) as X_file, open("%s/y_train.txt" % args.data) as y_file:
        X_text = X_file.read().splitlines()
        y_text = y_file.read().splitlines()
else:
    X_text, y_text = get_and_save_text(categories, args.data, test_size)

y = [y.split(",") for y in y_text]

logging.info("Collected %i examples across %i categories" % (len(X_text), len(categories)))

# Try different trainers (model pipelines)
# For each trainer, use randomized search CV
# Return CV score and refitted classifier
trainers = [
    MultinomialNBTrainer(),
    BernoulliNBTrainer(),
    LogisticRegressionTrainer(),
    RandomForestTrainer(),
    KNNTrainer(),
]

# Load or save binarizer
if os.path.isfile("%s/binarizer.pkl" % args.model):
    logging.info("Found binarizer in %s/binarizer.pkl" % args.model)
    binarizer = pickle.load(open("%s/binarizer.pkl" % args.model))
else:
    binarizer = MultiLabelBinarizer().fit(y)
    with open("%s/binarizer.pkl" % args.model, "w") as f:
        f.write(pickle.dumps(binarizer))
    logging.info("Saved binarizer to %s/binarizer.pkl" % args.model)

for trainer in trainers:
    trainer.train(X_text, binarizer.transform(y))

best_trainer = max(trainers, key=lambda m: m.score)
logging.info("Best model was %s with %f score" % (best_trainer.model_family, best_trainer.score))

with open(args.model, "w") as f:
    f.write(pickle.dumps(best_trainer.model.best_estimator_))

logging.info("Pickled and saved model to %s/model.pkl" % args.model)
