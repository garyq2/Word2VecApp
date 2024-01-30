import os
from dotenv import load_dotenv
load_dotenv('.env')
NLTK_PATH = os.getenv("NLTK_DATA")
TRAIN_PATH = os.getenv("TRAIN_PATH")
RAW_PATH = os.getenv("RAW_PATH")
PROCESSED_PATH=os.getenv("PROCESSED_PATH")
INTERIM_PATH = os.getenv("INTERIM_PATH")

from src.visualization.visualize import WordToVecApp 


if __name__ == '__main__':
    app: WordToVecApp = WordToVecApp()
    app.run()
    

    
    














