from fastapi import FastAPI
from pydantic import BaseModel
# from app.model.model import collaborative_filtering 
# from app.model.model import content_based_filtering
# from app.model.model import __version__ as model_version

from app.model.model import hybrid_based_recommendation_bayesian_approach, hybrid_based_recommendation_warp_approach
from random import randint
from app.model.model import collaborative_filtering 
from app.model.model import content_based_filtering
from app.model.model import __version__ as model_version

import uvicorn
#import traceback


app = FastAPI(title="Ohara-Bookshelf Model API", version="2.0.0", description="This is a simple API for the Bookshelf hybrid recommendation system ML Model")



class BookISBNInput(BaseModel):
    text: str

class RecommendationCountInput(BaseModel):
    count: int

class RecommendationOutput(BaseModel):
   #The recommentation output will be a list of book ISBNs
    books: list


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}



@app.post("/collaborative-filtering-recommendation", response_model = RecommendationOutput)
def collaborative_filtering_recommendation(ISBN: BookISBNInput, NUMBER: RecommendationCountInput):
    #Get collaborative filtering recommendations
    books_collaborative_filtering = collaborative_filtering(ISBN.text, NUMBER.count)
    #Convert the strings to lists
    books_collaborative_filtering = [books_collaborative_filtering]
    #Flatten the books list
    books_collaborative_filtering = [item for sublist in books_collaborative_filtering for item in sublist]
    #Remove duplicates
    books_collaborative_filtering = list(dict.fromkeys(books_collaborative_filtering))
    #Return the list
    return {"books": books_collaborative_filtering}


@app.post("/content-based-recommendation", response_model = RecommendationOutput)
def content_based_recommendation(ISBN: BookISBNInput, NUMBER: RecommendationCountInput):
    #Get content based filtering recommendations
    books_content_based_filtering = content_based_filtering(ISBN.text, NUMBER.count)
    #Convert the strings to lists
    books_content_based_filtering = [books_content_based_filtering]
    #Flatten the books list
    books_content_based_filtering = [item for sublist in books_content_based_filtering for item in sublist]
    #Remove duplicates
    books_content_based_filtering = list(dict.fromkeys(books_content_based_filtering))
    #Return the list
    return {"books": books_content_based_filtering}



@app.post("/hybrid-bayesian-recommendation", response_model=RecommendationOutput)
def hybrid_based_recommendation_api(ISBN: BookISBNInput, NUMBER: RecommendationCountInput):
    try:
        books = hybrid_based_recommendation_bayesian_approach(ISBN.text, NUMBER.count)
        return {"books": books}
    except Exception as e:
        return {"error": str(e)}
    

@app.post("/hybrid-warp-recommendation", response_model=RecommendationOutput)
def hybrid_based_recommendation_api(ISBN: BookISBNInput, NUMBER: RecommendationCountInput):
    try:
        books = hybrid_based_recommendation_warp_approach(ISBN.text, NUMBER.count)
        return {"books": books}
    except Exception as e:
        return {"error": str(e)}

@app.post("/hybrid-recommendation", response_model = RecommendationOutput)
def hybrid_recommendation(ISBN: BookISBNInput, NUMBER: RecommendationCountInput):

    # Use randomly one of the two hybrid approaches
    # hybrid_based_recommendation_bayesian_approach(ISBN.text, NUMBER.count)
    # hybrid_based_recommendation_warp_approach(ISBN.text, NUMBER.count)

    random_number = randint(1, 2)
    if random_number == 1:
        books = hybrid_based_recommendation_bayesian_approach(ISBN.text, NUMBER.count)
    else:
        books = hybrid_based_recommendation_warp_approach(ISBN.text, NUMBER.count)

    return {"books": books}

# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=4000)

