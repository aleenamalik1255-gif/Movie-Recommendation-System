import requests

url = "http://127.0.0.1:5000/predict"

# Example input (replace with real feature values from your dataset)
data = {
    "input": [10, 3.5, 1.2, 20, 4.0, 0.5, 1995, 2020, 6, 2, 0.5, 1, 100]
}

response = requests.post(url, json=data)
print(response.json())
