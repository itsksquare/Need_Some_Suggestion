import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

credits_df = pd.read_csv("https://raw.githubusercontent.com/itsksquare/datasets/main/movie/credits.csv?token=ASCPTSVEGT4MWSJJ2B7ESSDBZM6ZI")
movies_df = pd.read_csv("https://raw.githubusercontent.com/itsksquare/datasets/main/movie/movies.csv?token=ASCPTSUCUTXIYJPTDXQQWFDBZM63U")

credits_df.columns = ['id','tittle','cast','crew']
movies_df = movies_df.merge(credits_df, on="id")

def emotionmovie(movies_df=movies_df):
    movies_df = movies_df["title"]
    movies_df = movies_df.sample(n=16)
    return movies_df.to_numpy()

def topmoviesfn():
    popularity = movies_df.sort_values("popularity", ascending=False)
    return popularity["title"].to_numpy()

tfidf = TfidfVectorizer(stop_words="english")
movies_df["overview"] = movies_df["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(movies_df["overview"])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:17]
    movies_indices = [ind[0] for ind in sim_scores]
    movies = movies_df["title"].iloc[movies_indices]
    return movies

features = ["cast", "crew", "keywords", "genres"]

for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)

movies_df[features].head(10)

def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]

        if len(names) > 3:
            names = names[:3]
        return names
    return []

movies_df["director"] = movies_df["crew"].apply(get_director)

features = ["cast", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(get_list)

movies_df[['title', 'cast', 'director', 'keywords', 'genres']].head()

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""

features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)

movies_df = movies_df.reset_index()
indices = pd.Series(movies_df.index, index=movies_df['title'])