#!/usr/bin/env python3
# ufc_predictor.py

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_and_clean(path):
    df = pd.read_csv(path)
    # Drop rows missing key values
    df = df.dropna(subset=['wins', 'losses', 'weight_in_kg', 'date_of_birth'])
    # Compute age
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df = df.dropna(subset=['date_of_birth'])
    df['age'] = (datetime.now() - df['date_of_birth']).dt.days / 365.25
    return df

def simulate_matches(df, n=5000, random_state=42):
    np.random.seed(random_state)
    pairs = np.random.choice(df.index, size=(n,2))
    rows = []
    for i,j in pairs:
        f1, f2 = df.loc[i], df.loc[j]
        winner = 0 if f1['wins'] - f1['losses'] > f2['wins'] - f2['losses'] else 1
        rows.append({
            'age_diff': f1.age - f2.age,
            'winloss_diff': (f1.wins - f1.losses) - (f2.wins - f2.losses),
            'weight_diff': f1.weight_in_kg - f2.weight_in_kg,
            'winner': winner
        })
    return pd.DataFrame(rows)

def train_and_eval(df):
    X = df[['age_diff', 'winloss_diff', 'weight_diff']]
    y = df['winner']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    # Feature importance
    importances = clf.feature_importances_
    for feat, imp in zip(X.columns, importances):
        print(f"{feat}: {imp:.3f}")
    return clf

def main():
    df = load_and_clean('ufc_fighters_statistics.csv')
    fights = simulate_matches(df, n=5000)
    print("Sample fights:\n", fights.head())
    model = train_and_eval(fights)

if __name__ == '__main__':
    main()
