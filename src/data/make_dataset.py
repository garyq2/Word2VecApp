import os
import shutil
import random
from collections import Counter
from typing import List, Optional, Iterator, Tuple
from gensim import utils
import re
from dotenv import load_dotenv
load_dotenv('.env')
NLTK_PATH = os.getenv("NLTK_DATA")

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


class GensimCorpus:
    """An iterator that yields sentences (lists of str)."""
    
    def __init__(self, 
                 train_path: str,
                 raw_path: str,
                 processed_path: str,
                 interim_path: str,
                 user_filenames: Optional[List[str]]= None,
                 stoplist: Optional[set] = None):
        self.train_path = os.path.join(train_path)
        self.user_filenames = user_filenames
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.interim_path = interim_path
        self.randomize=True
        self.min_word_count = 2
        self.stoplist = stoplist if stoplist is not None else set(['the', 'of', 'and', 'for', 'in', 'a', 'to', 'on', 'with', 'is', 'are', 'its', 'an', 'are'])
        
    def __iter__(self) -> Iterator[Tuple[int, int]]:
        if self.user_filenames==None:
            for _file in self._get_default_file_paths():
                yield from self._read_and_preprocess_file(_file)
        else:
            for _file in os.listdir(self.raw_path):
                print("_file: ", _file)
                raw_file = os.path.join(self.raw_path,_file)
                print("raw_file: ", raw_file)
                yield from self._read_and_preprocess_file(raw_file)
                self._move_processed_file(raw_file)
                print("MOVED: ", raw_file)
    
    
    def _move_processed_file(self, filename):
        # Move the processed file to the processed directory
        dest = os.path.join(self.processed_path, os.path.basename(filename))
        shutil.move(filename, dest)
    
    def _get_default_file_paths(self) -> List[str]:
        train_files = [os.path.join(self.train_path, file) for file in os.listdir(self.train_path)]
        if self.randomize == True:
            random.shuffle(train_files)
        return train_files
    
    def _read_and_preprocess_file(self, 
                                  file_path: str) -> Iterator[List[str]]:
        word_counter = Counter()

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Tokenize into sentences using NLTK's sent_tokenize
                sentences = sent_tokenize(line)
                for sentence in sentences:
                    words = re.findall(r'[^\W\d_]+', sentence.lower())
                    words = [word for word in words if word not in self.stoplist]
                    word_counter.update(words)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Tokenize into sentences using NLTK's sent_tokenize
                sentences = sent_tokenize(line)
                for sentence in sentences:
                    words = re.findall(r'[^\W\d_]+', sentence.lower())
                    words = [word for word in words if word not in self.stoplist and word_counter[word] > self.min_word_count]
                    yield words
    
    
if __name__ == '__main__':
    from dotenv import load_dotenv
    
    load_dotenv(".env")
    TRAIN_PATH = os.getenv("TRAIN_PATH")
    RAW_PATH = os.getenv("RAW_PATH")
    PROCESSED_PATH=os.getenv("PROCESSED_PATH")
    NLTK_PATH = os.getenv("NLTK_DATA")
    INTERIM_PATH = os.getenv("INTERIM_PATH")
    
    
    corpus = GensimCorpus(TRAIN_PATH, RAW_PATH, PROCESSED_PATH, INTERIM_PATH)
    for ind, sentence in enumerate(corpus):
        print(sentence)
        if ind>10:
            break
