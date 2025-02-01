
# Movie Recommendation System

Recommender systems are essential for websites or online stores with millions of items, helping users find what they're looking for by narrowing down their choices. These systems are widely used by companies like Netflix and Amazon to suggest content tailored to individual users.

This project focuses on building a **Collaborative Filtering-based Movie Recommendation System**. The goal is to predict how a user would rate a movie they haven’t watched yet. The model will be evaluated using metrics like **RMSE (Root Mean Squared Error)** to minimize the difference between predicted and actual ratings.

## Table of Contents

1. [Project Setup](#project-setup)
2. [Data Collection](#data-collection)
3. [Data Cleaning](#data-cleaning)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Train and Test Splitting](#train-and-test-splitting)
6. [Model Building & Fitting](#model-building--fitting)
7. [Generate Recommendations](#generate-recommendations)
8. [Visualization](#visualization)
9. [Conclusion](#conclusion)

---

## Project Setup

To get started, you'll need to install the necessary libraries:

```bash
pip install scikit-surprise


Import the necessary libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
```

## Data Collection

Load and prepare the datasets containing the movie ratings and movie information:

```python
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
```

## Data Cleaning

We clean the data by checking for missing values, duplicates, and inspecting the dataset's general structure:

```python
# Checking for duplicates and missing values
ratings.isna().sum()
movies.isna().sum()
```

We also inspect the number of unique users and movies:

```python
total_users = ratings["userId"].nunique()
total_movies = movies["movieId"].nunique()
```

## Exploratory Data Analysis (EDA)

### Distribution of Ratings

We visualize the distribution of movie ratings:

```python
sns.countplot(x="rating", data=ratings, palette="viridis")
```

### Distribution of Genres

We analyze and visualize the distribution of genres:

```python
genres_series = movies["genres"].str.split("|").explode()
unique_genres = genres_series.value_counts()
sns.barplot(x=unique_genres.index, y=unique_genres.values, palette="viridis")
```

## Train and Test Splitting

We prepare the dataset for the **scikit-surprise** library and split it into training and testing sets:

```python
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
```

## Model Building & Fitting

We use **Singular Value Decomposition (SVD)** for collaborative filtering and train the model on the training set:

```python
model = SVD()
model.fit(trainset)
```

We make predictions on the test set and evaluate the model's performance using RMSE:

```python
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
```

## Generate Recommendations

A function is created to generate top movie recommendations for a user based on predicted ratings for movies the user hasn't rated yet:

```python
def get_movie_recommendations(user_id, num_recommendations=5):
    movie_ids = ratings['movieId'].unique()
    rated_movie_ids = ratings[ratings['userId'] == user_id]['movieId']
    movies_to_predict = [movie_id for movie_id in movie_ids if movie_id not in rated_movie_ids.values]
    
    predictions = [model.predict(user_id, movie_id) for movie_id in movies_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_predictions = predictions[:num_recommendations]
    recommended_movie_ids = [pred.iid for pred in top_predictions]
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    
    recommendations = pd.DataFrame({
        'movieId': recommended_movies['movieId'],
        'title': recommended_movies['title'],
        'estimated_rating': [pred.est for pred in top_predictions]
    })
    return recommendations
```

Example usage:

```python
recommendations = get_movie_recommendations(user_id=13000, num_recommendations=15)
```

## Visualization

### Top 10 Most Rated Movies

We visualize the top 10 most rated movies based on the number of ratings:

```python
top_10_most_rated = ratings["movieId"].value_counts().head(10)
top_10_most_rated_movies = movies[movies["movieId"].isin(top_10_most_rated.index)].copy()
top_10_most_rated_movies["num_ratings"] = top_10_most_rated.values

sns.barplot(x="num_ratings", y="title", data=top_10_most_rated_movies, palette="viridis")
```

## Conclusion

In this project, we:

- Explored the importance of **Recommendation Systems** and implemented a **Collaborative Filtering** technique using **SVD**.
- Predicted ratings for movies a user might enjoy based on their past behavior.
- Evaluated the model using the **RMSE** metric to assess prediction accuracy.
- Visualized the distribution of ratings and genres, as well as the most rated movies.

There is potential for further improvements in the system by experimenting with different algorithms and tuning model parameters.

---

## Future Improvements

- Experimenting with **Matrix Factorization** and **Deep Learning** techniques.
- Using **Hybrid Recommender Systems** that combine collaborative filtering with content-based methods.
- Tuning the model to improve accuracy.
- Incorporating additional features like movie metadata (e.g., director, actors) to enhance recommendations.

```
