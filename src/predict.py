
import pandas as pd
import joblib
import subprocess

def predict_diet(input_data):


    # Convert labels to integer codes for the ML model
    restriction_map = {"None": 0, "Low_Sugar": 1, "Vegetarian": 2}
    allergy_map = {"None": 0, "Peanuts": 1, "Lactose": 2}
    cuisine_map = {"Indian": 0, "Chinese": 1, "Italian": 2, "Mexican": 3}

    model_input = input_data.copy()
    model_input['Dietary_Restrictions'] = restriction_map.get(input_data['Dietary_Restrictions'], 0)
    model_input['Allergies'] = allergy_map.get(input_data['Allergies'], 0)
    model_input['Preferred_Cuisine'] = cuisine_map.get(input_data['Preferred_Cuisine'], 0)

    # Load model
    model = joblib.load("models/diet_classifier.pkl")
    input_df = pd.DataFrame([model_input])
    input_df = input_df.drop(columns=["restriction_label", "allergy_label", "cuisine_label"], errors="ignore")
    prediction = model.predict(input_df)[0]

    meal_plan = generate_meal_plan(prediction, input_data)

    return prediction, meal_plan
def generate_meal_plan(diet_type, input_data):
    prompt = f"""
Create a one-day personalized meal plan for a {input_data['Age']}-year-old {'male' if input_data['Gender'] == 1 else 'female'}.

Patient Profile:
- Diet Type: {diet_type}
- BMI: {input_data['BMI']}
- Weight: {input_data['Weight_kg']} kg
- Height: {input_data['Height_cm']} cm
- Disease Type Code: {input_data['Disease_Type']}
- Severity Level: {input_data['Severity']}
- Physical Activity Level: {input_data['Physical_Activity_Level']}
- Daily Caloric Intake: {input_data['Daily_Caloric_Intake']} kcal
- Cholesterol: {input_data['Cholesterol_mg/dL']} mg/dL
- Blood Pressure: {input_data['Blood_Pressure_mmHg']} mmHg
- Glucose: {input_data['Glucose_mg/dL']} mg/dL
- Nutrient Imbalance Score: {input_data['Dietary_Nutrient_Imbalance_Score']}
- Weekly Exercise Hours: {input_data['Weekly_Exercise_Hours']}
- Diet Adherence Level: {input_data['Adherence_to_Diet_Plan']}%
- Dietary Restriction: {input_data.get('restriction_label', 'None')}
- Allergy: {input_data.get('allergy_label', 'None')}
- Preferred Cuisine: {input_data.get('cuisine_label', 'General')}

Instructions:
Generate a realistic and nutritionally balanced meal plan including:
- Breakfast
- Lunch
- Dinner
- Two healthy snacks
- One clear health tip

Strictly follow the dietary restrictions and avoid allergens.
Keep the meal plan practical, easy to prepare, and aligned with the patient’s activity level and health profile.
    """

    try:

        print('prompt', prompt)
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
        'Dietary_Restrictions': 'Vegetarian',
        'Allergies': 'None',
        'Preferred_Cuisine': 'Indian',
        'Weekly_Exercise_Hours': 3.0,
        'Adherence_to_Diet_Plan': 80.0,
        'Dietary_Nutrient_Imbalance_Score': 2.5
    }

    diet_type, plan = predict_diet(sample_input)
    print(sample_input)
    print(f"\nPredicted Diet Recommendation: {diet_type}\n")
    print("Suggested Meal Plan:\n")
    print(plan)
