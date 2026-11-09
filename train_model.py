import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Fetch UCI Student Performance dataset (ID 320)
print("Downloading UCI Student Performance Dataset...")
student = fetch_ucirepo(id=320)

# Combine student features and target
df = pd.concat([student.data.features, student.data.targets], axis=1)

# Save Mathematics records as CSV
df.to_csv("student_data.csv", index=False)
print("Saved student_data.csv. Shape:", df.shape)

# 2. Preprocess & Target Definition
df = df.drop_duplicates()
# Target: 1 = At risk (G3 < 10), 0 = Not at risk (G3 >= 10)
y = (df["G3"] < 10).astype(int)

# Exclude G3 from features to avoid data leakage
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols)
])

# 3. Split & Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# 4. Model Evaluation & Export
predictions = model.predict(X_test)
print("\n--- Model Evaluation ---")
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0
))

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl successfully!")
