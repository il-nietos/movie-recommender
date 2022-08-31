#---------------------------------#
# this module 
# 1.reads in the data, transforms it to matrix form
# 2. trains the NMF model, and saves is

#---------------------------------#
import pandas as pd
import pickle
from sklearn.decomposition import NMF
import numpy as np

from utils import fill_nans


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


def train_nmf(matrix):
    """
    Train the NMF model
    """
    nmf = NMF(n_components=20, max_iter=500, init='random', random_state=np.random.randint(5))
    
    # Fit the model
    nmf.fit(matrix)  
    # Save the model with pickle package
    binary = pickle.dumps(nmf)
    open('nmf_model_reduced_size.bin', 'wb').write(binary)
    

def nmf_model():
    """
    Load NMF model, return the model and movie weights
    """
    nmf = pickle.loads(open('nmf_model_reduced_size.bin', 'rb').read())
    Q = nmf.components_
    return nmf, Q


# Call functions
movies, movie_matrix = read_and_transform(NMF=True)
train_nmf(movie_matrix)
nmf, Q = nmf_model()

