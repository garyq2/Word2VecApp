import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv('.env')
NLTK_PATH = os.getenv("NLTK_DATA")
TRAIN_PATH = os.getenv("TRAIN_PATH")
RAW_PATH = os.getenv("RAW_PATH")
PROCESSED_PATH=os.getenv("PROCESSED_PATH")
INTERIM_PATH = os.getenv("INTERIM_PATH")


from src.data.make_dataset import GensimCorpus
from src.models.train_model import Word2VecModel

class WordToVecApp:
    def __init__(self) -> None:
        self.TITLE: str = 'Word2Vec Model'
        self.word2vec_model: Word2VecModel = None
        self.words: List[str] = []
        self.embeddings: List[List[float]] = []
        self.MAX_FILE_SIZE_MB: int = 10
        self.MAX_FILES: int = 5

    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        return filename.lower().endswith('.txt')
    
    @staticmethod
    def validate_mime_type(mime_type: str) -> bool:
        return mime_type == 'text/plain'
    
    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int) -> bool:
        max_size_bytes: int = max_size_mb * 1024 * 1024  # Convert MB to bytes
        return file_size <= max_size_bytes
    
    def show_data_upload(self, raw_path: str, max_file_size_mb: int, max_files: int) -> List[str]:
        uploaded_files: List[st.uploaded_file_manager.UploadedFile] = st.file_uploader("Choose up to 5 text files", accept_multiple_files=True)
        file_paths: List[str] = []
        for uploaded_file in uploaded_files:
            if len(file_paths) >= max_files:
                st.error(f"Maximum {max_files} files allowed.")
                break
            if uploaded_file is not None:
                if not self.validate_file_extension(uploaded_file.name):
                    st.error("Please upload a .txt file.")
                    continue

                if not self.validate_mime_type(uploaded_file.type):
                    st.error("Uploaded file type is not supported.")
                    continue

                if not self.validate_file_size(uploaded_file.size, max_file_size_mb):
                    st.error(f"Uploaded file size exceeds the maximum allowed size of {max_file_size_mb} MB.")
                    continue

                bytes_data: bytes = uploaded_file.getvalue()
                file_content: str = bytes_data.decode('utf-8')
                if not any(file_content.strip()):
                    st.error("Uploaded file is empty.")
                    continue

                file_path: str = os.path.join(raw_path, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(bytes_data)

                st.write(f"File '{uploaded_file.name}' saved successfully.")
                file_paths.append(file_path)  # Return file path for loading
        return file_paths
    
    @staticmethod
    @st.cache_data
    def load_data(file_paths: List[str]) -> Tuple[Word2VecModel, List[str], List[List[float]]]:
        if file_paths:
            _sentences: GensimCorpus = GensimCorpus(train_path=os.getenv("TRAIN_PATH"),
                                                    raw_path=os.getenv("RAW_PATH"),
                                                    processed_path=os.getenv("PROCESSED_PATH"),
                                                    interim_path=os.getenv("INTERIM_PATH"),
                                                    user_filenames=file_paths)
            word2vec_model = Word2VecModel()
            word2vec_model.train(sentences=_sentences)
            words, embeddings = word2vec_model.get_word_embeddings()
            return word2vec_model, words, embeddings
    
    
    def show_similar_words(self, word: str, word2vec_model) -> None:
        if word2vec_model is not None:
            similar_words: List[Tuple[str, float]] = word2vec_model.model.wv.most_similar(word)
            st.subheader(f"Words similar to '{word}':")
            for similar_word, similarity in similar_words:
                st.write(f"- {similar_word} (similarity: {similarity:.2f})")
        else:
            st.error("Please upload a file and train the Word2Vec model first.")

    def visualize_word_embeddings(self, words, embeddings) -> None:
        if embeddings:
            tsne: TSNE = TSNE(n_components=2, perplexity=20.0, random_state=42)
            embeddings_2d: np.ndarray = tsne.fit_transform(np.array(embeddings))
            df: pd.DataFrame = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
            df['word'] = words
            fig = px.scatter(df, x='x', y='y', text='word')
            fig.update_traces(marker=dict(size=8))
            fig.update_layout(title='Word Embeddings Visualization', xaxis_title='Dimension 1', yaxis_title='Dimension 2')
            st.plotly_chart(fig)
        else:
            st.warning("Word2Vec model has not training data.")

    def run(self) -> None:
        st.title(self.TITLE)

        if 'button_clicked' not in st.session_state:
            st.session_state.button_clicked = False
        
        file_paths: List[str] = self.show_data_upload(os.getenv("RAW_PATH"), self.MAX_FILE_SIZE_MB,  self.MAX_FILES)
        
        if st.button("Load Data"):
            st.session_state.model, st.session_state.words, st.session_state.embeddings = WordToVecApp.load_data(file_paths)
            st.session_state.button_clicked = True
        
        enter_word: str = st.text_input("Enter a word:")
        
        if enter_word and st.session_state.button_clicked:
            self.show_similar_words(enter_word, st.session_state.model)
            self.visualize_word_embeddings(st.session_state.words, st.session_state.embeddings)


if __name__ == "__main__":
    app: WordToVecApp = WordToVecApp()
    app.run()

