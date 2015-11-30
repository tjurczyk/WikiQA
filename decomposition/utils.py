import re


def get_sentence_words(sentence):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()

    return string.split()