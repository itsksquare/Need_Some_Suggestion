import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans 
df = pd.read_csv('https://raw.githubusercontent.com/itsksquare/datasets/main/songs/data.csv?token=ASCPTSQJZCCCNS3WGDB4CUTBZM65M')

def emotionsong(df=df):
    df = df["name"]
    df = df.sample(n=16)
    return df.to_numpy()

def topsongsfn():
    popularity = df.sort_values("popularity", ascending=False)
    return popularity["name"].to_numpy()

def normalize_column(col):
    max_d = df[col].max()
    min_d = df[col].min()
    df[col] = (df[col] - min_d)/(max_d - min_d)
    
num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num = df.select_dtypes(include=num_types)
        
for col in num.columns:
    normalize_column(col)
    
km = KMeans(n_clusters=25)
pred = km.fit_predict(num)
df['pred'] = pred
normalize_column('pred')

class Song_Recommender():
    def __init__(self, data):
        self.data_ = data
    
    def get_recommendations(self, song_name):
        rem_data = self.data_[self.data_.name.str.lower() != song_name.lower()]
        return rem_data['name'].head(16).tolist()

recommender = Song_Recommender(df)

def s_recommend(s_name, recommender = recommender):
    return recommender.get_recommendations(s_name)