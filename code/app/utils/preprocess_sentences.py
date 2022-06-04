"""
Functions to extract and process the sentences from a DBpedia abstract
Author: Fernando Casabán Blasco and Pablo Hernández Carrascosa
"""
import re

######################
# Sentence functions #
######################


def get_sentences(doc):
    """
    Get a list with the sentences of the input document (spacy).
    """
    sentences = []
    for sente in doc.sents:
        sentences.append(sente)
    return sentences


def clean_text(text):
    """
    Remove characters bounded by parentheses, always than there is no other parentheses inside.
    """
    text = re.sub("\([^()]+\) ", "", text)
    return text
