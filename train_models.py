import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

print("[INFO] Loading dataset 'netflix_titles.csv'...")
df = pd.read_csv('netflix_titles.csv')

df_clean = df.copy()
df_clean['target'] = (df_clean['type'] == 'Movie').astype(int)

df_clean['title_length'] = df_clean['title'].fillna('').apply(len)
df_clean['title_word_count'] = df_clean['title'].fillna('').apply(lambda x: len(x.split()))
df_clean['desc_length'] = df_clean['description'].fillna('').apply(len)
df_clean['desc_word_count'] = df_clean['description'].fillna('').apply(lambda x: len(x.split()))
df_clean['cast_count'] = df_clean['cast'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
df_clean['country_count'] = df_clean['country'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
df_clean['has_director'] = df_clean['director'].notnull().astype(int)
df_clean['is_us'] = df_clean['country'].fillna('').apply(lambda x: 1 if 'United States' in x else 0)
df_clean['is_india'] = df_clean['country'].fillna('').apply(lambda x: 1 if 'India' in x else 0)
df_clean['is_uk'] = df_clean['country'].fillna('').apply(lambda x: 1 if 'United Kingdom' in x else 0)

df_clean['date_added'] = pd.to_datetime(df_clean['date_added'].str.strip(), errors='coerce')
df_clean['year_added'] = df_clean['date_added'].dt.year.fillna(df_clean['release_year']).astype(int)
df_clean['month_added'] = df_clean['date_added'].dt.month.fillna(6).astype(int)

df_clean['is_drama'] = df_clean['listed_in'].fillna('').apply(lambda x: 1 if 'Dramas' in x else 0)
df_clean['is_comedy'] = df_clean['listed_in'].fillna('').apply(lambda x: 1 if 'Comedies' in x else 0)
df_clean['is_action'] = df_clean['listed_in'].fillna('').apply(lambda x: 1 if 'Action' in x else 0)
df_clean['is_doc'] = df_clean['listed_in'].fillna('').apply(lambda x: 1 if 'Docu' in x else 0)
df_clean['is_international'] = df_clean['listed_in'].fillna('').apply(lambda x: 1 if 'International' in x else 0)

feature_cols = [
    'release_year', 'year_added', 'month_added', 
    'title_length', 'title_word_count', 
    'desc_length', 'desc_word_count', 
    'cast_count', 'country_count', 'has_director',
    'is_us', 'is_india', 'is_uk',
    'is_drama', 'is_comedy', 'is_action', 'is_doc', 'is_international'
]

X = df_clean[feature_cols]
y = df_clean['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

pipelines = {
    "logistic_regression": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42, max_iter=1000))]),
    "decision_tree": Pipeline([('clf', DecisionTreeClassifier(random_state=42, max_depth=6))]),
    "knn": Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=7))]),
    "naive_bayes": Pipeline([('clf', GaussianNB())]),
    "support_vector_machine": Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=42, kernel='rbf'))])
}

os.makedirs('model', exist_ok=True)

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"[MODEL] {name:<22} -> Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    joblib.dump(pipe, f"model/{name}.joblib")

print("[SUCCESS] All models trained and saved to /model successfully!")
