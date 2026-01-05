import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print(">>> Titanic Survival Prediction Script <<<")

try:
    df = pd.read_csv("Titanic-Dataset.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Oops! 'Titanic-Dataset.csv' not found. Please check the file path.")
    exit()

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nCleaning data...")

df['Age'].fillna(df['Age'].median(), inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.drop('Cabin', axis=1, inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

print("Data cleaned! Here's a preview:")
print(df.head())

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Logistic Regression model (might take a sec)...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Training complete!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Did not Survive', 'Survived'])

print("\n=============================")
print("  MODEL PERFORMANCE RESULTS  ")
print("=============================")
print(f"Accuracy: {accuracy:.2%}")
print("\n(So the model is correct about ~"
      f"{accuracy:.0%} of the passengers in test data.)")

print("\n--- Classification Report ---")
print(report)

print("\nPassenger count by gender (after encoding):")
print(df.info())
print(df.head())
