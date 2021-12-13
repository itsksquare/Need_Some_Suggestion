from re import T
import numpy as np
import pandas as pd
from typing import List,Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

songs=pd.read_csv('recommender/songs/songdata.csv')

songs=songs.sample(n=5000).drop('link',axis=1).reset_index(drop=True)

songs['text']=songs['text'].str.replace(r'\n','')

tfidf=TfidfVectorizer(analyzer='word',stop_words='english')

lyrics_matrix=tfidf.fit_transform(songs['text'])

cosine_similarities=cosine_similarity(lyrics_matrix)

similarities={}

for i in range(len(cosine_similarities)):
    similar_indices=cosine_similarities[i].argsort()[:-50:-1]
    similarities[songs['song'].iloc[i]]=[(cosine_similarities[i][x],songs['song'][x],songs['artist'][x]) for x in similar_indices][1:]

class ContentBasedRecommender:
    def __init__(self,matrix):
        self.matrix_similar=matrix
    def _print_messages(self,song,recom_song):
        rec_items=len(recom_song)
        print(f'The {rec_items} recommeded songs for {song} are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_song[i][1]} by {recom_song[i][2]} with {round (recom_song[i][0],3)} similarity score")
            print("-----------------")

    def recommend(self,recommendation):
        song=recommendation['song']
        number_songs=recommendation['number_songs']
        recom_song=self.matrix_similar[song][:number_songs]
        self._print_messages(song=song,recom_song=recom_song)  

recommendations=ContentBasedRecommender(similarities)

recommendation={
    "song":songs['song'].iloc[10],
    "number_songs":4
}

recommendations.recommend(recommendation)

recommendation2={
    "song":songs['song'].iloc[120],
    "number_songs":4
}

recommendations.recommend(recommendation2)

