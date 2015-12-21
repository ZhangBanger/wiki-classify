import re

from gensim.corpora import wikicorpus


def filter_additional_items(text):
    text = re.sub(r"={2,}.*?={2,}", " ", text)
    text = text.replace("\n", " ")
    text = text.encode("utf-8")
    return text


def preprocess_wiki_text(wiki_text):
    text = wikicorpus.filter_wiki(wiki_text)
    text = filter_additional_items(text)
    return text
