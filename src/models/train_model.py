import os
import logging
from dotenv import load_dotenv
from typing import Iterator, Tuple, Union, List 
import streamlit as st

from gensim.models import Word2Vec
from src.data.make_dataset import GensimCorpus


class Word2VecModel:
    def __init__(self, 
                 vector_size: int = 100, 
                 window: int = 5, 
                 min_count: int = 1, 
                 sg: int = 0, 
                 epochs: int = 1, 
                 workers: int = 10):
        self.vector_size: int = vector_size
        self.window: int = window
        self.min_count: int = min_count
        self.sg: int = sg
        self.epochs: int = epochs
        self.workers: int = workers
        self.model: Word2Vec = self._create_model()
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    # st.cache_resource()
    def _create_model(self) -> Word2Vec:
        """
        Create and return a Word2Vec model.
        """
        return Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=self.workers
        )
    
    # st.cache_resource()
    def train(self, sentences: List[List[str]]) -> Word2Vec:
        """
        Train the Word2Vec model.

        Args:
        - sentences (List[List[str]]): List of tokenized sentences.

        Returns:
        - Word2Vec: Trained Word2Vec model.
        """
        self.logger.info("Training Model")
        self.model.build_vocab(sentences, update=False)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=self.epochs)
        self.logger.info("Training completed.")
        
    # st.cache_resource()
    def get_word_embeddings(self):
        words = list(self.model.wv.index_to_key)
        embeddings = [self.model.wv[word] for word in words]
        return words, embeddings
    
    # st.cache_resource()
    def get_model(self) -> Word2Vec:
        """
        Return the Word2Vec model.

        Returns:
        - Word2Vec: Trained Word2Vec model.
        """
        return self.model





##################################################################
if __name__ == '__main__':
    load_dotenv(".env")
    TRAIN_PATH = os.getenv("TRAIN_PATH")
    RAW_PATH = os.getenv("RAW_PATH")
    PROCESSED_PATH=os.getenv("PROCESSED_PATH")
    NLTK_PATH = os.getenv("NLTK_DATA")
    INTERIM_PATH = os.getenv("INTERIM_PATH")
    
    
    sentences = GensimCorpus(TRAIN_PATH, RAW_PATH, PROCESSED_PATH, INTERIM_PATH)
    model = Word2VecModel(workers=10)
    model.train(sentences=sentences)