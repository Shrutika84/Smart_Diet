# predict.py with Ollama-based dynamic meal generation

import pandas as pd
import joblib
import subprocess

def predict_diet(input_data):
    # Load trained model
    model = joblib.load("models/diet_classifier.pkl")

    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])

    # Predict diet type
    prediction = model.predict(input_df)[0]

    # Generate meal plan using Ollama
    age = input_data.get('Age', '30')
    gender = 'male' if input_data.get('Gender', 1) == 1 else 'female'
    meal_plan = generate_meal_plan(prediction, age, gender)

    return prediction, meal_plan

def generate_meal_plan(diet_type, age, gender):
    prompt = f"""
    Create a meal plan for a {age}-year-old {gender} following a {diet_type} diet.
    Include: breakfast, lunch, dinner, two snacks, and one health tip.
    Keep it practical, healthy, and easy to follow.
    """

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error generating meal plan: {e.stderr}"

# Example usage
if __name__ == "__main__":
    sample_input = {
        'Age': 35,
        'Gender': 1,
        'Weight_kg': 72.0,
        'Height_cm': 170,
        'BMI': 24.9,
        'Disease_Type': 1,
        'Severity': 1,
        'Physical_Activity_Level': 2,
        'Daily_Caloric_Intake': 2400,
        'Cholesterol_mg/dL': 180.0,
        'Blood_Pressure_mmHg': 120,
        'Glucose_mg/dL': 95.0,
        'Dietary_Restrictions': 0,
        'Allergies': 0,
        'Preferred_Cuisine': 2,
        'Weekly_Exercise_Hours': 3.0,
        'Adherence_to_Diet_Plan': 80.0,
        'Dietary_Nutrient_Imbalance_Score': 2.5
    }
    diet_type, plan = predict_diet(sample_input)
    print(f"\nPredicted Diet Recommendation: {diet_type}\n")
    print("Suggested Meal Plan:\n")
    print(plan)
