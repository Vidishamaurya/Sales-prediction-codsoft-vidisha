import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

print(df.head())
print(df.info())

df.columns = df.columns.str.strip()

df = df.dropna(subset=['Rating'])

df['Year'] = df['Year'].str.extract('(\d{4})').astype(float)

df['Duration'] = df['Duration'].str.replace(' min', '', regex=True)
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

df['Votes'] = df['Votes'].str.replace(',', '', regex=True)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

df.fillna(df.median(numeric_only=True), inplace=True)

encoder = LabelEncoder()
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col].astype(str))

plt.figure(figsize=(6,4))
sns.histplot(df['Rating'], bins=20)
plt.title("IMDB Rating Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

X = df[['Year', 'Duration', 'Votes']]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted IMDB Ratings")
plt.show()

sample_movie = pd.DataFrame({
    'Year': [2020],
    'Duration': [120],
    'Votes': [150000]
})

predicted_rating = model.predict(sample_movie)
print("\nPredicted IMDB Rating:", predicted_rating[0])
