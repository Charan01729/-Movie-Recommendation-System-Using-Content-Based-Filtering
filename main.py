from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from difflib import get_close_matches

# Load saved components
cv = pickle.load(open("cv_countvec.pkl", "rb"))
(indices, distances) = pickle.load(open("knn_indices_distances.pkl", "rb"))
new = pickle.load(open("movie_list_df.pkl", "rb"))

titles = new['title'].tolist()

app = FastAPI()

class MovieQuery(BaseModel):
    movie_name: str
    k: int = 5

def find_title(query):
    if query in titles:
        return query
    suggestions = get_close_matches(query, titles, n=5, cutoff=0.4)
    return suggestions[0] if suggestions else None

@app.get("/")
def home():
    return {"message": "Movie Recommendation API is running"}

@app.post("/recommend")
def recommend(data: MovieQuery):
    movie = data.movie_name
    k = data.k

    title = find_title(movie)
    if not title:
        return {"error": "Movie not found"}

    idx = new.index[new['title'] == title][0]
    rec_idx = indices[idx, 1:k+1]
    rec_titles = new.iloc[rec_idx]['title'].tolist()

    return {
        "input_movie": title,
        "recommendations": rec_titles
    }
