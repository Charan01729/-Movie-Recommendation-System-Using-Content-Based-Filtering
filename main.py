from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
from difflib import get_close_matches

# ----- Load Artifacts -----
cv = pickle.load(open("cv_countvec.pkl", "rb"))
(indices, distances) = pickle.load(open("knn_indices_distances.pkl", "rb"))
new = pickle.load(open("movie_list_df.pkl", "rb"))

titles = new["title"].tolist()

app = FastAPI(title="Movie Recommendation API")

templates = Jinja2Templates(directory="templates")

class MovieQuery(BaseModel):
    movie_name: str
    k: int = 5

def find_title(query: str):
    if query in titles:
        return query
    suggestions = get_close_matches(query, titles, n=5, cutoff=0.4)
    return suggestions[0] if suggestions else None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"message": "Movie Recommendation API is running"}

@app.post("/recommend")
def recommend_movie(data: MovieQuery):
    movie_query = data.movie_name
    k = data.k

    title = find_title(movie_query)
    if not title:
        return {"error": "Movie not found in dataset (or no close match)."}

    idx = new.index[new["title"] == title][0]
    rec_positions = indices[idx, 1 : k + 1]
    rec_titles = new.iloc[rec_positions]["title"].tolist()

    return {
        "input_movie": title,
        "recommendations": rec_titles,
    }
