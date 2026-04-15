#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_and_clean(path):
    """
    Step 1: Get the data ready. 
    We filter out incomplete profiles and calculate age from birthdates.
    """
    df = pd.read_csv(path)
    
    # We drop rows missing these columns because the model needs them to learn
    df = df.dropna(subset=['wins', 'losses', 'weight_in_kg', 'date_of_birth'])
    
    # Converting text dates to 'datetime' objects so we can do math with them
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df = df.dropna(subset=['date_of_birth'])
    
    # Calculate age by finding the difference between today and their birthday
    df['age'] = (datetime.now() - df['date_of_birth']).dt.days / 365.25
    return df

def simulate_matches(df, n=5000, random_state=42):
    """
    Step 2: Create a training dataset.
    Since we don't have a list of real past fights, we 'simulate' 5,000 matches
    by picking two random fighters and comparing their stats.
    """
    np.random.seed(random_state)
    pairs = np.random.choice(df.index, size=(n, 2))
    rows = []
    
    for i, j in pairs:
        f1, f2 = df.loc[i], df.loc[j]
        
        # We define the 'winner' as whoever has a better career record (Wins minus Losses)
        winner = 0 if (f1['wins'] - f1['losses']) > (f2['wins'] - f2['losses']) else 1
        
        # Engineering 'Reach': If reach is missing, we use height as a logical substitute
        reach1 = f1['reach_in_cm'] if pd.notnull(f1['reach_in_cm']) else f1['height_cm']
        reach2 = f2['reach_in_cm'] if pd.notnull(f2['reach_in_cm']) else f2['height_cm']

        # We store the DIFFERENCES between fighters (this is what the AI looks at)
        rows.append({
            'age_diff': f1.age - f2.age,
            'winloss_diff': (f1.wins - f1.losses) - (f2.wins - f2.losses),
            'weight_diff': f1.weight_in_kg - f2.weight_in_kg,
            'reach_diff': reach1 - reach2,
            'winner': winner
        })
    return pd.DataFrame(rows)

def train_and_eval(df):
    """
    Step 3: Train the AI.
    We use a Random Forest (a collection of decision trees) to find patterns 
    in the differences we calculated above.
    """
    # X = Input features; y = The target we want to predict (the winner)
    X = df[['age_diff', 'winloss_diff', 'weight_diff', 'reach_diff']]
    y = df['winner']
    
    # Split data: 80% to train the AI, 20% to test if it actually learned correctly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Initialize and train the Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    clf.fit(X_train, y_train)
    
    # Test the model on the 20% it hasn't seen yet
    y_pred = clf.predict(X_test)
    
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    
    # Create a visual chart to show which stats matter most to the AI
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
    feat_importances.sort_values().plot(kind='barh', color='#2ecc71')
    plt.title('Which Stats Predict the Winner?')
    plt.xlabel('Importance (How much the AI relies on this stat)')
    plt.tight_layout()
    plt.savefig('feature_importance.png') 
    
    return clf

def main():
    # The 'engine' that runs the whole process in order
    print("Running UFC Prediction Model...")
    df = load_and_clean('ufc_fighters_statistics.csv')
    fights = simulate_matches(df, n=5000)
    train_and_eval(fights)
    print("Success! Check your folder for 'feature_importance.png'.")

if __name__ == '__main__':
    main()
if __name__ == '__main__':
    main()
