
import pandas as pd
import numpy as np
#from scipy.spatial import distance
from utils import id_to_title, fuzzy_title_to_id_only, fill_nans
from read_train import read_and_transform
import pickle 
from sklearn.decomposition import NMF
import heapq

# Load models
#nmf = pickle.loads(open('models/nmf_model_reduced_size.bin', 'rb').read())

user_input = [1,2,5,6,7]
# movieIds: 

def read_and_transform(NMF =True):
    """
    Read and transform the csv files
    """
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    ratings.set_index('movieId', inplace=True)
    ratings.drop('timestamp', axis=1, inplace=True)

    # merge the two dataframes
    movies_ratings = ratings.merge(movies, how='inner',on='movieId')
    movies_ratings.drop(columns=['genres', 'title'], axis=1, inplace=True)

    # Convert to matrix form where user in index, movies in first row, ratings as values
    movie_matrix = movies_ratings.pivot(index="userId", columns="movieId", values="rating")

    # Filter out movies that have been watched by less than 3% of users (<5/ 610 users)
    movie_matrix = movie_matrix.dropna(thresh=movie_matrix.shape[0]*0.03,how='all', axis=1)
   

    # Define how null values filled based on model 
    if NMF:
        movie_matrix = fill_nans(movie_matrix)
    else:
        movie_matrix = movie_matrix.fillna(0)
    
    return movies, movie_matrix


movies, movie_matrix = read_and_transform()
movieids = movie_matrix.columns.to_numpy()

#print('====movies_matrix=====', movie_matrix.head(2))

def nmf_model():
    """
    Load NMF model, return the model and movie weights
    """
    nmf = pickle.loads(open('models/nmf_model_reduced_size.bin', 'rb').read())
    Q = nmf.components_
    return nmf, Q

nmf, Q = nmf_model()
#print('Q', Q)

def prepare_new_user(user_input, matrix=movie_matrix, NMF=True): # this will be called in application.py
    '''
    input: user input
    output: new user array
    '''
    # from user input(movie ids) to dictionary - all movies rates 5 (favorite movies asked)
    user_dict = dict.fromkeys(user_input, 5)

    # append user info to matrix
    if NMF:
        new_user = np.full_like(matrix.columns, 2.5, dtype=float)
    else:
        new_user = np.zeros_like(matrix.columns)
    
    for index, item in enumerate(matrix.columns):
        if user_dict.get(item):  # returns the value for key if the key is in the dictionary
            new_user[index] = user_dict[item]
    
    user_df = pd.DataFrame([new_user],index=['new_user'], columns = matrix.columns)
    return user_df

user_df = prepare_new_user(user_input=[1, 2, 5, 6, 7], matrix=movie_matrix, NMF=True)
#print('=====user_df=======', user_df)

def nmf_predict(new_user=prepare_new_user(user_input)):
    '''
    NMF model 
    '''
    nmf = pickle.loads(open('models/nmf_model_reduced_size.bin', 'rb').read())
    P_user = nmf.transform(new_user) 
    # new user is a dataframe
    R_user = np.dot(P_user, Q).flatten() # create R matrix and ensure one dimensionl output
    moviedict = {}
    for A, B in zip(movieids, R_user): moviedict[A] = B # place movies ids (keys) and predicted ratings (valeues) in dictionary
    top_movie_id = heapq.nlargest(5, moviedict, key=moviedict.get) # retrieve movie ids of top 5 highest ratings
    print(top_movie_id)
    
    return id_to_title(movies, top_movie_id)


top_movies = nmf_predict(user_df)
#print('=======top nmf movies=======', top_movies)

def recommend_random(query, movies, k=5):
    """
    Dummy recommender that recommends a list of random movies. Ignores the actual query.
    """
    random_movie_ids =  movies['movieId'].sample(k).to_list()
    return id_to_title(movies, random_movie_ids)


random_movies = recommend_random(query=None, movies=movies, k=5)
#print('=======top random movies=======', random_movies)

