import pickle

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import nltk

class NotInCorpusError(Exception):
    pass

class RedditModel:

    def __init__(self):
        self.model = self._load_model()

        self.country_fixes = {}

    def get_similarity(self, word1, word2):
        
        word1 = self._clean_word(word1)
        word2 = self._clean_word(word2)
        
        try:
            return self.model.similarity(word1, word2)
        except KeyError as e:
            raise NotInCorpusError from e

    def get_nearest(self, word, n):
        """
        Return n nearest words for given word

        Params:
            word (str) 
            n (int)         number of nearest words
        
        Returns:
            (list(str))     nearest words
        
        """
        word = self._clean_word(word)
        
        try:
            return self.model.wv.most_similar(word, topn=n)
        except KeyError as e:
           raise NotInCorpusError from e
    
    def get_nearest_algebra(self, positive, negative, n):
        
        positive = [self._clean_word(w) for w in positive]
        negative = [self._clean_word(w) for w in negative]

        try:
            return self.model.most_similar(positive=positive, negative=negative, topn=n)
        except KeyError as e:
            raise NotInCorpusError from e

    def _train_model(self):
        raise NotImplementedError

    def _load_model(self, fname='model.bin'):
        """
        Return cached model

        Params:
            fname (str)     path to cached model
        """
        return KeyedVectors.load_word2vec_format(fname, binary=True)

    def _clean_word(self, word):
        """
        Return a preprocessed version of searched word.

        Manually fix ambiguous country names

        Params:
            word (str)
        
        Returns
            (str) word
        """
        cleaned_word = word.strip().lower()
        
        return self.country_fixes[cleaned_word] if cleaned_word in self.country_fixes.keys() else cleaned_word

if __name__ == '__main__':

    model = RedditModel()

    print(model.get_nearest_algebra(positive=['King, woman'], negative=['man'], n=10))


        