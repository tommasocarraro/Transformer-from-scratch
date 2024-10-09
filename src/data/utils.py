import re


def tokenize_(sentence):
    """
    It tokenizes the given sentence and returns a list of tokens.

    :param sentence: sentence to be tokenized
    :return: the tokenized sentence
    """
    return sentence.strip().split()


def process_sentence(sentence):
    """
    Process sentence so it is compatible with how vocabularies are built.
    :param sentence: sentence to be processed
    :return: processed sentence
    """
    sentence = sentence.lower()
    sentence = re.sub(r'([.,!?])', r' \1 ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.strip()
    return sentence
