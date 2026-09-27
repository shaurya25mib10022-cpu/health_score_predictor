"""
train_model.py
Trains a robust Machine Learning model (RandomForestRegressor) for overall health score prediction
based on clinical health metrics, lifestyle indicators, and demographic data.
The dataset is synthesized using realistic multivariate distributions grounded in 
epidemiological research (NHANES, Framingham Heart Study, and AHA Life's Essential 8).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def generate_health_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    
    # Demographics
    age = np.random.randint(18, 85, size=n_samples)
    gender_code = np.random.choice([0, 1], size=n_samples) # 0: Female, 1: Male
    
    # Anthropometrics
    height = np.where(gender_code == 1, 
                      np.random.normal(176, 7, size=n_samples), 
                      np.random.normal(163, 6, size=n_samples))
    height = np.clip(height, 140, 210)
    
    # Weight correlated slightly with age and lifestyle
    base_bmi = np.random.normal(25.5, 4.5, size=n_samples)
    base_bmi = np.clip(base_bmi, 16.0, 48.0)
    weight = base_bmi * ((height / 100.0) ** 2)
    
    # Vitals
    # Blood pressure correlates with age and BMI
    systolic_bp = 100 + (age * 0.35) + (base_bmi * 0.8) + np.random.normal(0, 10, size=n_samples)
    systolic_bp = np.clip(systolic_bp, 90, 200)
    
    diastolic_bp = 65 + (age * 0.15) + (base_bmi * 0.5) + np.random.normal(0, 7, size=n_samples)
    diastolic_bp = np.clip(diastolic_bp, 55, 125)
    
    resting_hr = np.random.normal(70, 10, size=n_samples)
    resting_hr = np.clip(resting_hr, 45, 115)
    
    # Fasting glucose (mg/dL) - normal 70-99, prediabetes 100-125, diabetes 126+
    glucose = 75 + (age * 0.2) + (base_bmi * 0.7) + np.random.normal(0, 12, size=n_samples)
    glucose = np.clip(glucose, 65, 250)
    
    # Total cholesterol (mg/dL)
    cholesterol = 160 + (age * 0.5) + (base_bmi * 0.6) + np.random.normal(0, 25, size=n_samples)
    cholesterol = np.clip(cholesterol, 120, 320)
    
    # Lifestyle factors
    activity_mins = np.random.exponential(scale=140, size=n_samples)
    activity_mins = np.clip(activity_mins, 0, 700)
    
    daily_steps = (activity_mins * 25) + np.random.normal(4500, 1800, size=n_samples)
    daily_steps = np.clip(daily_steps, 1000, 25000)
    
    sleep_hours = np.random.normal(7.1, 1.2, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 3.5, 11.0)
    
    sleep_quality = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.1, 0.2, 0.35, 0.25, 0.1])
    
    water_liters = np.random.normal(2.1, 0.7, size=n_samples)
    water_liters = np.clip(water_liters, 0.5, 5.0)
    
    diet_quality = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.12, 0.22, 0.36, 0.2, 0.1])
    
    # Risk behaviors
    # Smoking: 0: Never, 1: Former, 2: Occasional, 3: Daily
    smoking_code = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.55, 0.20, 0.10, 0.15])
    
    # Alcohol: 0: None, 1: Light, 2: Moderate, 3: Heavy
    alcohol_code = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.35, 0.35, 0.20, 0.10])
    
    stress_level = np.random.randint(1, 11, size=n_samples) # 1-10
    screen_time = np.random.normal(6.5, 2.5, size=n_samples)
    screen_time = np.clip(screen_time, 1.0, 16.0)
    
    # Medical history (0 or 1)
    family_cvd = np.random.choice([0, 1], size=n_samples, p=[0.72, 0.28])
    family_diabetes = np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])
    family_hypertension = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])

    # Ground truth clinical composite calculation for target health score (0 to 100)
    # Based on AHA Life's Essential 8 scoring components
    
    # 1. BMI Component (0 - 20 pts)
    bmi_score = np.where(base_bmi < 18.5, 12,
                np.where(base_bmi <= 24.9, 20,
                np.where(base_bmi <= 29.9, 15,
                np.where(base_bmi <= 34.9, 9, 4))))
    
    # 2. Blood Pressure Component (0 - 20 pts)
    bp_penalty = np.maximum(0, (systolic_bp - 115) / 4.0) + np.maximum(0, (diastolic_bp - 75) / 2.5)
    bp_score = np.clip(20 - bp_penalty, 2, 20)
    
    # 3. Glycemic / Metabolic Component (0 - 15 pts)
    glucose_penalty = np.where(glucose < 100, 0,
                      np.where(glucose < 126, (glucose - 99) * 0.3,
                               8 + (glucose - 125) * 0.1))
    chol_penalty = np.maximum(0, (cholesterol - 190) * 0.05)
    metabolic_score = np.clip(15 - glucose_penalty - chol_penalty, 1, 15)
    
    # 4. Physical Activity & Fitness Component (0 - 15 pts)
    act_score = np.clip((activity_mins / 150.0) * 10 + (daily_steps / 10000.0) * 5, 1, 15)
    
    # 5. Sleep & Recovery Component (0 - 12 pts)
    sleep_dur_score = np.where((sleep_hours >= 7.0) & (sleep_hours <= 8.5), 6,
                      np.where((sleep_hours >= 6.0) & (sleep_hours <= 9.5), 4, 1.5))
    sleep_qual_score = (sleep_quality / 5.0) * 6
    sleep_score = np.clip(sleep_dur_score + sleep_qual_score, 1, 12)
    
    # 6. Nutrition & Hydration (0 - 10 pts)
    diet_pts = (diet_quality / 5.0) * 7.5
    water_pts = np.clip(water_liters / 2.5, 0.2, 1.0) * 2.5
    diet_score = diet_pts + water_pts
    
    # 7. Stress & Habits Deductions (Up to -25 pts)
    stress_penalty = (stress_level / 10.0) * 5.0
    smoking_penalty = smoking_code * 3.5 # Up to -10.5
    alcohol_penalty = np.where(alcohol_code == 3, 5.0, np.where(alcohol_code == 2, 2.0, 0))
    family_penalty = (family_cvd * 1.5) + (family_diabetes * 1.0) + (family_hypertension * 1.0)
    screen_penalty = np.maximum(0, (screen_time - 7) * 0.4)
    
    deductions = stress_penalty + smoking_penalty + alcohol_penalty + family_penalty + screen_penalty
    
    raw_health_score = bmi_score + bp_score + metabolic_score + act_score + sleep_score + diet_score - deductions
    
    # Add slight realistic stochastic variation (human individuality)
    health_score = raw_health_score + np.random.normal(0, 2.5, size=n_samples)
    health_score = np.clip(health_score, 15.0, 99.0)

    df = pd.DataFrame({
        'age': age,
        'gender_code': gender_code,
        'height': np.round(height, 1),
        'weight': np.round(weight, 1),
        'bmi': np.round(base_bmi, 1),
        'systolic_bp': np.round(systolic_bp, 1),
        'diastolic_bp': np.round(diastolic_bp, 1),
        'resting_hr': np.round(resting_hr, 1),
        'glucose': np.round(glucose, 1),
        'cholesterol': np.round(cholesterol, 1),
        'activity_mins': np.round(activity_mins, 1),
        'daily_steps': np.round(daily_steps, 0),
        'sleep_hours': np.round(sleep_hours, 1),
        'sleep_quality': sleep_quality,
        'water_liters': np.round(water_liters, 1),
        'diet_quality': diet_quality,
        'smoking_code': smoking_code,
        'alcohol_code': alcohol_code,
        'stress_level': stress_level,
        'screen_time': np.round(screen_time, 1),
        'family_cvd': family_cvd,
        'family_diabetes': family_diabetes,
        'family_hypertension': family_hypertension,
        'health_score': np.round(health_score, 1)
    })
    
    return df

def train_and_export_model():
    print("Generating representative clinical dataset...")
    df = generate_health_dataset(n_samples=6000, random_state=42)
    
    feature_cols = [
        'age', 'gender_code', 'height', 'weight', 'bmi',
        'systolic_bp', 'diastolic_bp', 'resting_hr',
        'glucose', 'cholesterol', 'activity_mins', 'daily_steps',
        'sleep_hours', 'sleep_quality', 'water_liters', 'diet_quality',
        'smoking_code', 'alcohol_code', 'stress_level', 'screen_time',
        'family_cvd', 'family_diabetes', 'family_hypertension'
    ]
    
    X = df[feature_cols]
    y = df['health_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    print(f"Training RandomForestRegressor on {len(X_train)} samples...")
    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    
    print(f"Model evaluation:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R^2 Score: {r2:.4f}")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'health_model.joblib')
    
    payload = {
        'model': model,
        'feature_cols': feature_cols,
        'metrics': {
            'rmse': rmse,
            'r2': r2
        }
    }
    joblib.dump(payload, model_path)
    print(f"Model saved successfully to {model_path}!")

if __name__ == '__main__':
    train_and_export_model()
