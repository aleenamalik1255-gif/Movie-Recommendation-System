import requests

url = "http://127.0.0.1:5000/predict"

# Example input (replace with real feature values from your dataset)
data = {
    "input": [
    2,        # user_count
    3.0,      # user_mean
    2.824,    # user_std
    368,      # movie_count
    4.2065,   # movie_mean
    0.9224,   # movie_std
    1980.0,   # movie_year
    1997,     # ts_year
    12,       # ts_month
    3,        # ts_dayofweek
    0.79347,  # user_movie_mean_diff
    0,        # userId
    172,      # movieId 
    ]
}

response = requests.post(url, json=data)
print(response.json())
