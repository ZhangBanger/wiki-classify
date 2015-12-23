import argparse
import os
import pickle

from wiki_utils import get_article_by_title

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="path to save model", default="%s/model.pkl" % os.getcwd())
parser.add_argument("-t", "--title", help="plain english title as appears")
args = parser.parse_args()

model = pickle.load(open(args.model))
classes = model.steps[1][1].classes_

article_text = get_article_by_title(args.title)

print("Prediction for %s :" % args.title)

top_prediction = model.predict([article_text])[0]

prediction_dist = model.predict_proba([article_text])

prediction_dist_labeled = dict(zip(classes, prediction_dist[0]))
print("Best prediction: %s" % top_prediction)
print("Prediction distribution")
for category, probability in prediction_dist_labeled.iteritems():
    print ("%s -> %.2f" % (category, probability))
