import pickle
import re
from pathlib import Path
#import traceback
# import streamlit as st
import pandas as pd




__version__ = "2.0.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


#TODO: CONTENT BASED RECOMMENDATION MODEL PART:
#Load the books model:

books_dictionary = pickle.load(open(f"{BASE_DIR}/final_books-dataset-{__version__}.pkl", "rb"))

books_model = pd.DataFrame(books_dictionary)

#load the model
content_based_simularity_model = pickle.load(open(f"{BASE_DIR}/content-based-similarity-{__version__}.pkl", "rb"))

#TODO: TEST THE MODEL
# print("Books Model: ",books_model.head())

# print("Content Based Simularity Model: ",content_based_simularity_model)

#Recommendation function
def content_based_filtering(isbn, number_of_books):
    # if the ISBN is not provided
    if isbn is None:
        return 'ISBN not provided.'
    
    #if the book is not in the dataframe
    if isbn not in books_model['ISBN'].unique():
        #return 'Book not in the dataset.'
        return []
    else:
        #get the index of the book in the dataframe
        book_index = books_model[books_model['ISBN'] == isbn].index[0]
        #get the list of similar books
        distances = content_based_simularity_model[book_index]
        #sort the list
        books_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:number_of_books+1]
    
        #return all the recommended books titles
        #recommended_books = [books_model.iloc[book[0]].title for book in books_list]
         #return all the recommended books ISBNs
        recommended_books = [books_model.iloc[book[0]].ISBN for book in books_list]
        return recommended_books
        

#TODO: TCOLLABORATIVE FILTERING MODEL PART:


#Load the books, ratings and users model:
books_isbn_title_rating = pickle.load(open(f"{BASE_DIR}/book_isbn_title_user_rating_model-{__version__}.pkl", "rb"))

books_isbn_title_rating_model = pd.DataFrame(books_isbn_title_rating)
#print("Books Model: ",books_isbn_title_user_rating_model.head())

#load the recommendation model
collaborative_filtering_simularity_model = pickle.load(open(f"{BASE_DIR}/collaborative-filtering-simularity-{__version__}.pkl", "rb"))

#TODO: TEST THE MODEL
# print("Books Model: ",books_isbn_title_rating_model.head())

# print("Collaborative Filtering Simularity Model: ",collaborative_filtering_simularity_model)


def collaborative_filtering(isbn, number_of_books):

    # if the ISBN is not provided
    if isbn is None:
        return 'ISBN not provided.'
    
    # if the book is not in the dataframe
    if isbn not in books_isbn_title_rating_model.index.get_level_values(0):
        #return 'Book not in the dataset.'
        #here return nothing:
        return []
    else:

        book_index = books_isbn_title_rating_model.index.get_level_values(0).get_loc(isbn)
        distances = list(enumerate(collaborative_filtering_simularity_model[book_index]))
        sorted_distances = sorted(distances, key=lambda x:x[1], reverse=True)
        similar_books = sorted_distances[1:number_of_books+1]
        #Simular books isbn:
        similar_books_isbn = [books_isbn_title_rating_model.index.get_level_values(0)[i[0]] for i in similar_books]
       
        return similar_books_isbn



#TODO: HYBRID RECOMMENDATION MODEL PART:


# TODO: BAYESIAN APPROACH
def hybrid_based_recommendation_bayesian_approach(isbn, number_of_books):
    # Get recommendations from collaborative filtering model
    collab_recs = collaborative_filtering(isbn, number_of_books)
    # Get recommendations from content-based filtering model
    content_recs = content_based_filtering(isbn, number_of_books)

    #If collaborative filtering model returns nothing:
    if len(collab_recs) == 0:
        return content_recs
    elif len(content_recs) == 0:
        return collab_recs
    
    elif(len(collab_recs) == 0 and len(content_recs) == 0):
        return []
    # Combine the recommendations using a Bayesian network
    else:

        recommendations = []
        for collab_rec in collab_recs:
            for content_rec in content_recs:
                if collab_rec == content_rec:
                    # Recommendations from both models agree, so we can be more confident in this recommendation
                    recommendations.append(collab_rec)
        if len(recommendations) < number_of_books:
            # If we don't have enough recommendations yet, add the remaining items from either model
            recommendations.extend(collab_recs[len(recommendations):])
            recommendations.extend(content_recs[len(recommendations):])
    
        # Return the top 'number_of_books' recommendations
        return recommendations[:number_of_books]


# TODO: WARP APPROACH
def hybrid_based_recommendation_warp_approach(isbn, number_of_books):
    # Get recommendations from collaborative filtering model
    collab_recs = collaborative_filtering(isbn, number_of_books)
  
    # Get recommendations from content-based filtering model
    content_recs = content_based_filtering(isbn, number_of_books)
    
    #If collaborative filtering model returns nothing:
    if len(collab_recs) == 0:
        return content_recs
    elif len(content_recs) == 0:
        return collab_recs
    
    elif(len(collab_recs) == 0 and len(content_recs) == 0):
        return []
    # Combine the recommendations using a Bayesian network
    else:
        # Combine the recommendations using WARP loss
        recommendations = []
        for i, collab_rec in enumerate(collab_recs):
            for j, content_rec in enumerate(content_recs):
                if collab_rec == content_rec:
                    # Recommendations from both models agree, so we can assign a higher weight to this recommendation
                    recommendations.append((collab_rec, i + j))
        recommendations.sort(key=lambda x: x[1])
    
        # Return the top 'number_of_books' recommendations
        return [rec[0] for rec in recommendations][:number_of_books]