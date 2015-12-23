import argparse
import logging
import os
import pickle

from trainer import MultinomialNBTrainer, BernoulliNBTrainer, LinearModelTrainer, RandomForestTrainer
from wiki_utils import get_and_save_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

categories = [category.rstrip() for category in open("%s/categories.txt" % os.getcwd())]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="dir to save or load existing X.txt, y.txt", default=os.getcwd())
parser.add_argument("-m", "--model", help="path to save model", default="%s/model.pkl" % os.getcwd())
args = parser.parse_args()

logging.info("Collecting data for all categories")

# Initialize and collect data
X_text, y_text = [], []
test_size = 0.2

# Load X_train.txt and y_train.txt or fetch train and test sets
if os.path.isfile("%s/X_train.txt" % args.data) and os.path.isfile("%s/y_train.txt" % args.data):
    logging.info("Found data in %s: " % args.data)
    with open("%s/X_train.txt" % args.data) as X_file, open("%s/y_train.txt" % args.data) as y_file:
        X_text = X_file.readlines()
        y_text = y_file.readlines()
else:
    X_text, y_text = get_and_save_text(categories, args.data, test_size)

logging.info("Collected %i examples across %i categories" % (len(X_text), len(categories)))

# Try different trainers (model pipelines)
# For each trainer, use randomized search CV
# Return CV score and refitted classifier
trainers = [
    MultinomialNBTrainer(),
    BernoulliNBTrainer(),
    LinearModelTrainer(),
    RandomForestTrainer(),
]

for trainer in trainers:
    trainer.train(X_text, y_text)

best_trainer = max(trainers, key=lambda m: m.score)
logging.info("Best model was %s with %f score" % (best_trainer.model_family, best_trainer.score))

with open(args.model, "w") as f:
    f.write(pickle.dumps(best_trainer.model.best_estimator_))

logging.info("Pickled and saved model to %s" % args.model)
