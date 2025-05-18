# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv("Medicine_Details.csv")
df = df[['Composition', 'Uses', 'Side_effects', 'Manufacturer', 'Average Review %', 'Poor Review %', 'Excellent Review %']]
df.dropna(inplace=True)

X = df.drop('Excellent Review %', axis=1)
y = df['Excellent Review %']

# Preprocessing and model pipeline
preprocessor = ColumnTransformer([
    ('composition', TfidfVectorizer(max_features=100), 'Composition'),
    ('uses', TfidfVectorizer(max_features=100), 'Uses'),
    ('side_effects', TfidfVectorizer(max_features=100), 'Side_effects'),
    ('manufacturer', OneHotEncoder(handle_unknown='ignore'), ['Manufacturer']),
    ('numeric', 'passthrough', ['Average Review %', 'Poor Review %'])
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
])

# Train-test split and fitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'medicine_review_model.pkl')
print("✅ Model saved successfully!")

# Evaluate on test data
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"📈 R² Score (Test): {r2:.4f}")
print(f"📉 RMSE (Test): {rmse:.2f}")

# 🔍 Print training R² for comparison
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
print(f"🧠 R² Score (Train): {train_r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"\n🔁 5-Fold Cross-Validated R² Scores: {cv_scores}")
print(f"📊 Mean R² Score from Cross-Validation: {cv_scores.mean():.4f}")

# =========================
# 📊 Feature Importance Plot
# =========================
# Access components
rf = model.named_steps['regressor']
preprocessor = model.named_steps['preprocessor']

# Extract feature names
tfidf_comp = preprocessor.named_transformers_['composition'].get_feature_names_out()
tfidf_uses = preprocessor.named_transformers_['uses'].get_feature_names_out()
tfidf_side = preprocessor.named_transformers_['side_effects'].get_feature_names_out()
manufacturer = preprocessor.named_transformers_['manufacturer'].get_feature_names_out(preprocessor.transformers_[3][2])
numeric = ['Average Review %', 'Poor Review %']

# Combine feature names in the same order they go into the model
feature_names = list(tfidf_comp) + list(tfidf_uses) + list(tfidf_side) + list(manufacturer) + numeric

# Match feature importances
importances = rf.feature_importances_
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(20)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feat_df['Feature'], feat_df['Importance'], color='skyblue')
plt.xlabel("Importance")
plt.title("Top 20 Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
