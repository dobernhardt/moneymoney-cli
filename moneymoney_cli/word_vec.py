from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import gensim.downloader as api


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="glove-wiki-gigaword-50"):
        self.model_name = model_name
        self.word2vec = api.load(model_name)
        self.dim = self.word2vec.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._get_sentence_vector(sentence) for sentence in X])

    def _get_sentence_vector(self, sentence):
        words = sentence.split()
        word_vectors = [self.word2vec[word] for word in words if word in self.word2vec]
        if len(word_vectors) == 0:
            return np.zeros(self.dim)
        return np.mean(word_vectors, axis=0)
