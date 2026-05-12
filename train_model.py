import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("dataset.csv")

X = data[['area','bedrooms','bathrooms']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model Trained Successfully")
