import nltk
import numpy as np
from nltk.corpus import stopwords, brown
from nltk.cluster.util import cosine_distance
from operator import itemgetter

nltk.download('stopwords')
nltk.download('brown')
# np.seterr(divide='ignore', invalid='ignore')


def run_page_rank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)

    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P


def get_sentence_similarity(sentence_1, sentence_2, stop_words):
    if stop_words is None:
        stop_words = []

    sent1a = []
    for word in sentence_1:
        sent1a.append(word.lower())

    sent2a = []
    for word in sentence_2:
        sent2a.append(word.lower())

    all_words = list(set(sent1a + sent2a))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in sentence_1:
        if word in stop_words:
            continue
        vector1[all_words.index(word.lower())] += 1

    for word in sentence_2:
        if word in stop_words:
            continue
        vector2[all_words.index(word.lower())] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    num_sentences = len(sentences)
    s = np.zeros(num_sentences, num_sentences)

    for index1 in range(num_sentences):
        for index2 in range(num_sentences):
            if index1 == index2:
                continue
            s[index1][index2] = get_sentence_similarity(sentences[index1], sentences[index2], stop_words)

    for index3 in range(len(s)):
        if s[index3].sum() == 0:
            continue
        s[index3] /= s[index3].sum()

    return s


def main():
    stop_words = stopwords.words('english')

    # sentences = nltk.sent_tokenize(text)
    sentences = brown.sents('ca01')

    # TODO: import text file and tokenize it into sentences,
    #  then tokenize those sentences into words
    #  ex: [[u'The', u'Fulton', u'County', u'Grand', u'Jury', u'said', u'Friday' ...], ...]

    s = build_similarity_matrix(sentences, stop_words)

    # TODO: fix pagerank implementation
    sentence_ranks = run_page_rank(s)

    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    selected_sentences = sorted(ranked_sentence_indexes[:20])
    summary = itemgetter(*selected_sentences)(sentences)

    for sentence in summary:
        print(' '.join(sentence))


main()
