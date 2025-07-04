
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load processed data
df = pd.read_csv("data/cleaned_data.csv")

# Separate features and target
X = df.drop(columns=['Diet_Recommendation'])
y = df['Diet_Recommendation']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/diet_classifier.pkl")
print("\nModel saved to models/diet_classifier.pkl")
