"""
train_model.py
──────────────
Run this ONCE before launching the app to pre-train the Random Forest
model and save it to disk. The app will load it instantly on startup
instead of retraining every time.

Usage:
    python train_model.py
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

POLLUTANT_COLS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
                  'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']

print("Loading data...")
df = pd.read_csv("cleaned_data.csv")

X = df[POLLUTANT_COLS].fillna(df[POLLUTANT_COLS].median())
le = LabelEncoder()
y = le.fit_transform(df['AQI_Bucket'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest (120 trees)...")
model = RandomForestClassifier(
    n_estimators=120,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump({'model': model, 'label_encoder': le, 'accuracy': acc}, 'model.pkl')
print("\n✅ Saved to model.pkl — you can now launch the app.")
