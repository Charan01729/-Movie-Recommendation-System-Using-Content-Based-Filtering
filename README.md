# Movie Recommendation System Using Content-Based Filtering

This project implements a content-based movie recommendation system using Python, Pandas, and Scikit-Learn. The system recommends movies based on their metadata, such as genres, cast, and keywords, using vectorization and cosine similarity.

## Technologies Used

*   Python
*   Pandas
*   Scikit-Learn

## Features

*   Recommends similar movies based on user-selected input titles.
*   Analyzes movie metadata (genres, cast, keywords) to generate personalized recommendations.
*   Uses text-based similarity scores (cosine similarity) for recommendations.

## Data

The dataset used for this project consists of two CSV files:

*   `tmdb_5000_movies.csv`: Contains information about movies, including title, genres, keywords, and overview.
*   `tmdb_5000_credits.csv`: Contains information about movie cast and crew, including director and lead actors.

These files are merged based on the movie title to create a comprehensive dataset for the recommendation system.

## Methodology

The recommendation system uses a content-based filtering approach:

1.  **Data Preprocessing**: The relevant columns (`genres`, `keywords`, `cast`, `crew`, `overview`) are extracted from the merged dataset. Missing values are handled by dropping rows. The text data in these columns is cleaned and transformed into a format suitable for vectorization. This includes converting JSON strings to lists, extracting names, and removing spaces within names to create single tokens.
2.  **Tag Creation**: A new column called `tags` is created by combining the processed text data from the `overview`, `genres`, `keywords`, `cast`, and `crew` columns.
3.  **Vectorization**: The `tags` are converted into a numerical representation using the `CountVectorizer`. This creates a sparse matrix where each row represents a movie and each column represents a word in the vocabulary.
4.  **Similarity Calculation**: The `NearestNeighbors` algorithm is used to find movies that are similar to each other based on their vector representation. Cosine similarity is used as the distance metric. The neighbors for all movies are precomputed for faster lookup.
5.  **Recommendation**: Given a movie title as input, the system finds the corresponding movie in the dataset, retrieves its precomputed neighbors, and returns the titles of the top-k similar movies. Fuzzy matching is used to handle potential typos in the input movie title.

## Performance Metrics

The model was evaluated on a sample dataset, achieving the following metrics:

*   Precision@5: 0.9154
*   Recall@5: 0.0027
*   MAP@5: 0.8890
*   NDCG@5: 0.9185

Note: The relevance for evaluation is defined by genre overlap between the query movie and the recommended movie.

## Files

*   `tmdb_5000_movies.csv`: The movie dataset.
*   `tmdb_5000_credits.csv`: The movie credits dataset.
*   `requirements.txt`: A file listing the necessary Python packages. (Optional, but good practice for reproducibility)
*   `movie_recommendation.ipynb`: The Jupyter Notebook containing the code for the recommendation system. (Assuming this is the name of the notebook)
