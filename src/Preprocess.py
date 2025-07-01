# preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_and_encode_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # Drop Patient_ID if present
    df.drop(columns=['Patient_ID'], inplace=True, errors='ignore')

    # Fill missing categorical values with mode
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Fill missing numerical values with mean
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != 'Diet_Recommendation':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

    return df, label_encoders

if __name__ == "__main__":
    clean_and_encode_data("data/diet_recommendations_dataset.csv", "data/cleaned_data.csv")
