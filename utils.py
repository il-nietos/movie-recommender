import pandas as pd
from thefuzz import process # a library that helps matching strings 

# Read in movies csv data to pandas dataframe  

movies = pd.read_csv('movies.csv')
movies.set_index('movieId', drop=False, inplace=True) # have movieId both as column and as index 


# Function to retrieve title based on movieid
def id_to_title(movies_df, movie_ids):
    '''
    transforms moveId to title
    '''
    title = []
    for movie_id in movie_ids:
        title.append(movies_df.query(f'movieId =={movie_id}')['title'].values[0])
    
    return title

# from fuzzy title to clean title 
def fuzzy_title_to_id(movies_df, title_from_user):
    '''
    input user movie title
    returns movie title, fuzz score and movieId (in this order)
    '''
    clean_title =[]

    for title in title_from_user:
        title.append(movies_df.query(f'movieId =={movie_id}')['title'].values[0])
    
    title_full = process.extract(f'{title_from_user}', movies_df['title'], limit=1)
    return title_full


# from fuzzy title to clean title 
def fuzzy_title_to_id(movies_df, title_from_user):
    '''
    input user movie title
    returns movie title, fuzz score and movieId (in this order)
    '''
    title_full = process.extract(f'{title_from_user}', movies_df['title'], limit=1)
    return title_full


def fuzzy_title_to_clean_title(movies_df, title_from_user):
    '''
    input user movie title
    returns movie title, fuzz score and movieId (in this order)
    '''
    clean_titles =[]
    for title in title_from_user:
        title_full = process.extract(f'{title}', movies_df['title'], limit=1)[0][0]
        clean_titles.append(title_full)
    return clean_titles


def fuzzy_title_to_id_only(movies_df, title_from_user):
    '''
    input user movie title
    returns movie title, fuzz score and movieId (in this order)
    '''
    id_only = process.extract(f'{title_from_user}', movies_df['title'], limit=1)[0][2]
    return id_only

def fill_nans(df):
    '''
    Fill in nan values with mean rating of other users for the particular movie 
    '''
    for movie in df.columns:
        df[movie].fillna(round(df[movie].mean(), 1), inplace=True)
    return df
