import logging
import re
from urllib import quote

import requests
from gensim.corpora import wikicorpus
from sklearn.cross_validation import train_test_split

wiki_api_base = 'https://en.wikipedia.org/w/api.php?'

# Placeholders for category name (correctly formatted) and continue param
# Hardcoded batch size and pages only (no subcategories)
category_query = wiki_api_base + \
                 'action=query&format=json&list=categorymembers' + \
                 '&cmlimit=50&cmtype=page&cmtitle=Category:%s&cmcontinue=%s'
# Placeholders for page ids
article_query = wiki_api_base + 'action=query&format=json&prop=revisions&rvprop=content&pageids=%s'

title_query = wiki_api_base + 'action=query&format=json&prop=revisions&rvprop=content&titles=%s'

logging.getLogger("requests").setLevel(logging.WARNING)


class NoArticleError(Exception):
    pass


def get_and_save_text(categories, data_dir, test_size=0.2):
    # Collect pairs of text, labels
    x_text = []
    y_text = []
    category_map = {}
    article_map = {}

    # Get text examples for all categories
    # Write results to file
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

            page_ids = [page['pageid'] for page in category_resp['query']['categorymembers']]

            old_pages = [page_id for page_id in page_ids if page_id in category_map]
            for page_id in old_pages:
                category_map[page_id].add(category)

            new_pages = [page_id for page_id in page_ids if page_id not in category_map]
            for page_id in new_pages:
                category_map[page_id] = {category}

            # Get all new article text in batch
            article_resp = requests.get(article_query % "|".join(map(str, new_pages))).json()
            articles = [(int(pid), page['revisions'][0]['*']) for pid, page in article_resp['query']['pages'].items()]
            for page_id, text in articles:
                preprocessed_text = preprocess_wiki_text(text)
                article_map[page_id] = preprocessed_text

    for page_id, text in article_map.iteritems():
        x_text.append(text)
        y_text.append(",".join(category_map[page_id]))

    # Can no longer stratify due to multi-label
    x_train, x_test, y_train, y_test = train_test_split(x_text, y_text, test_size=test_size)

    with open("%s/X_train.txt" % data_dir, "w") as f:
        f.write('\n'.join(x_train) + "\n")

    with open("%s/X_test.txt" % data_dir, "w") as f:
        f.write('\n'.join(x_test) + "\n")

    with open("%s/y_train.txt" % data_dir, "w") as f:
        f.write('\n'.join(y_train) + "\n")

    with open("%s/y_test.txt" % data_dir, "w") as f:
        f.write('\n'.join(y_test) + "\n")

    logging.info("Saved data to %s as X_train.txt, X_test.txt, y_train.txt, and y_test.txt" % data_dir)

    return x_train, y_train


def get_article_by_title(title):
    quoted_title = quote(title)
    article_resp = requests.get(title_query % quoted_title).json()

    try:
        article_raw = article_resp['query']['pages'].items()[0][1]['revisions'][0]["*"]
    except KeyError:
        raise NoArticleError("Title '%s' does not exist" % title)

    return preprocess_wiki_text(article_raw)


def filter_additional_items(text):
    text = re.sub(r"={2,}.*?={2,}", " ", text)
    text = text.replace("\n", " ")
    text = text.encode("utf-8")
    return text


def preprocess_wiki_text(wiki_text):
    text = wikicorpus.filter_wiki(wiki_text)
    text = filter_additional_items(text)
    return text
