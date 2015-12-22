import logging
import re

import requests
from gensim.corpora import wikicorpus

wiki_api_base = 'https://en.wikipedia.org/w/api.php?'

# Placeholders for category name (correctly formatted) and continue param
# Hardcoded batch size and pages only (no subcategories)
category_query = wiki_api_base + \
                 'action=query&format=json&list=categorymembers' + \
                 '&cmlimit=50&cmtype=page&cmtitle=Category:%s&cmcontinue=%s'
# Placeholders for page ids
article_query = wiki_api_base + 'action=query&format=json&prop=revisions&rvprop=content&pageids=%s'

logging.getLogger("requests").setLevel(logging.WARNING)


def get_and_save_text(categories, data_dir):
    # Collect pairs of text, label
    x_text = []
    y_text = []

    # Get text examples for all categories
    # Write results to file
    with open("%s/X.txt" % data_dir, "w") as X_file, open("%s/y.txt" % data_dir, "w") as y_file:
        for category in categories:
            continue_param = '-||'

            logging.info("Collecting data for category: %s" % category)

            batch_count = 0

            # Get data in batches per wiki api limit
            while continue_param:
                logging.debug("Processing batch: %i" % batch_count)
                batch_count += 1

                # Continue query and update continue_param
                category_resp = requests.get(category_query % (category, continue_param)).json()
                if category_resp.get('continue'):
                    continue_param = category_resp['continue'].get('cmcontinue')
                else:
                    # Finished
                    continue_param = None

                # This will bomb out if query is empty, which is what we want
                page_ids = [page['pageid'] for page in category_resp['query']['categorymembers']]

                # Get all article text in batch and append to scraped_text
                article_resp = requests.get(article_query % "|".join(map(str, page_ids))).json()
                articles = [page['revisions'][0]['*'] for _, page in article_resp['query']['pages'].iteritems()]
                for text in articles:
                    preprocessed_text = preprocess_wiki_text(text)
                    x_text.append(preprocessed_text)
                    y_text.append(category)

                    X_file.write(preprocessed_text + "\n")
                    y_file.write(category + "\n")

    logging.info("Saved data to %s as X.txt and y.txt" % data_dir)

    return x_text, y_text


def filter_additional_items(text):
    text = re.sub(r"={2,}.*?={2,}", " ", text)
    text = text.replace("\n", " ")
    text = text.encode("utf-8")
    return text


def preprocess_wiki_text(wiki_text):
    text = wikicorpus.filter_wiki(wiki_text)
    text = filter_additional_items(text)
    return text
