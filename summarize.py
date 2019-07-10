import nltk
nltk.download('gutenberg')
import numpy as np
# np.seterr(divide='ignore', invalid='ignore')
from nltk.corpus import stopwords, brown
from nltk.corpus import gutenberg
from nltk.cluster.util import cosine_distance
from operator import itemgetter 



def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P

def sentence_similarity(sent1, sent2, stop):
    if stop is None:
        stop = []

    sent1 = []
    for w in sent1:
        sent1.append(w.lower())

    sent2 = []
    for w in sent2:
        sent2.append(w.lower())

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1: 
        if w in stop:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2: 
        if w in stop:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop):
    noSentences = len(sentences)
    s = np.zeros((noSentences, noSentences))

    for index1 in range(noSentences):
        for index2 in range(noSentences):
            if index1 == index2:
                continue
            s[index1][index2] = sentence_similarity(sentences[index1], sentences[index2], stop )

    for index3 in range(len(s)):
        if s[index3].sum()==0:
            continue
        s[index3] /= s[index3].sum()

    return s

def main():
    # text = gutenberg.words('carroll-alice.txt')
    stop_words = stopwords.words('english')

    # sentences = nltk.sent_tokenize(text)
    #sentences = brown.sents('ca01')

    #TODO import text file and tokenize it into sentences, 
    # then tokenzse those sentences into words
    # Ex: [[u'The', u'Fulton', u'County', u'Grand', u'Jury', u'said', u'Friday', u'an', u'investigation', u'of', u"Atlanta's", u'recent', u'primary', u'election', u'produced', u'``', u'no', u'evidence', u"''", u'that', u'any', u'irregularities', u'took', u'place', u'.'], [u'The', u'jury', u'further', u'said', u'in', u'term-end', u'presentments', u'that', u'the', u'City', u'Executive', u'Committee', u',', u'which', u'had', u'over-all', u'charge', u'of', u'the', u'election', u',', u'``', u'deserves', u'the', u'praise', u'and', u'thanks', u'of', u'the', u'City', u'of', u'Atlanta', u"''", u'for', u'the', u'manner', u'in', u'which', u'the', u'election', u'was', u'conducted', u'.'], ...]

    #S = build_similarity_matrix(sentences, stop_words)

    #TODO: fix pagerank implementation
    #sentence_ranks = pagerank(S) 

    # Sort the sentence ranks
    # ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    # selected_sentences = sorted(ranked_sentence_indexes[:20])
    # summary = itemgetter(*selected_sentences)(sentences)

    # for sentence in summary:
    #     print(' '.join(sentence))

main()








    


