from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

features = [
    #"HomeGoalsLast5",
    #"AwayGoalsLast5",
    "HS", "AS",
    "HST", "AST",
    "HC", "AC",
    "HF", "AF",
]

df = pd.read_csv('Data/Processed_Data/premstat2025-26.csv')

df = df.dropna()

X = df[features]
y = df["FTR"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
