import argparse
import logging
import os
import pickle

from trainer import MultinomialNBTrainer, BernoulliNBTrainer, LinearModelTrainer, RandomForestTrainer
from wiki_utils import get_and_save_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

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
parser.add_argument("-m", "--model", help="path to save model pipeline", default="%s/model.pkl" % os.getcwd())
args = parser.parse_args()

logging.info("Collecting data for all categories")

# Initialize and collect data
X_text, y_text = [], []

if os.path.isfile("%s/X.txt" % args.data) and os.path.isfile("%s/y.txt" % args.data):
    logging.info("Found data in %s: " % args.data)
    with open("%s/X.txt" % args.data) as X_file, open("%s/y.txt" % args.data) as y_file:
        X_text = X_file.readlines()
        y_text = y_file.readlines()
else:
    X_text, y_text = get_and_save_text(categories, args.data)

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

trained_models = [trainer.train(X_text, y_text) for trainer in trainers]

best_trainer = max(trained_models, key=lambda m: m.score)
logging.info("Best model was %s with %f score" % (best_trainer.model_family, best_trainer.score))

print(best_trainer.model)

with open(args.model, "w") as model_file:
    model_file.write(pickle.dumps(best_trainer.model))

logging.info("Pickled and saved model to %s" % args.model)
