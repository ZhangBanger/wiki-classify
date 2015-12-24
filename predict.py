import argparse
import os
import pickle

from wiki_utils import get_article_by_title, NoArticleError

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="directory with model and binarizer", default=os.getcwd())
parser.add_argument("-t", "--title", help="plain english title as appears")
args = parser.parse_args()

model = pickle.load(open("%s/model.pkl" % args.model))
binarizer = pickle.load(open("%s/binarizer.pkl" % args.model))
classes = binarizer.classes_

article_text = None

try:
    article_text = get_article_by_title(args.title)
except NoArticleError as e:
    print e
    exit(1)

top_prediction = binarizer.inverse_transform(model.predict([article_text]))[0]
prediction_dist = model.predict_proba([article_text])
prediction_dist_labeled = dict(zip(classes, prediction_dist[0]))

print("Tags for '%s': %s" % (args.title, ",".join(top_prediction) if top_prediction else ""))
print("Tag probabilities")
for category, probability in sorted(prediction_dist_labeled.iteritems(), key=lambda item: item[1], reverse=True):
    print ("%s -> %.2f" % (category, probability))
