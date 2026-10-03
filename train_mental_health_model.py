"""
train_mental_health_model.py
Trains a dual-output Machine Learning model (RandomForestRegressor) for:
1. Mental Health Score (0 - 100: Higher indicates greater psychological flourishing & resilience)
2. Stress Score (0 - 100: Higher indicates higher perceived stress & psychological strain)

Ground truth calculations are synthesized based on validated clinical psychometric frameworks:
- PSS-10 (Perceived Stress Scale by Cohen et al.)
- PHQ-9 (Patient Health Questionnaire - Depression scale)
- GAD-7 (Generalized Anxiety Disorder scale)
- WHO-5 (Well-Being Index)
- Maslach Burnout Inventory dimensions
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

FEATURE_COLS = [
    'q_anhedonia',           # 0-4: Little interest or pleasure
    'q_depressed',           # 0-4: Feeling down, depressed, hopeless
    'q_optimism',            # 0-4: Feeling cheerful, optimistic (reverse distress)
    'q_overwhelmed',         # 0-4: Difficulties piling up so high cannot overcome
    'q_uncontrollable',      # 0-4: Unable to control important things
    'q_coping_confidence',   # 0-4: Confident in handling personal problems (reverse distress)
    'q_anxious',             # 0-4: Feeling nervous, anxious, on edge
    'q_worry',               # 0-4: Inability to stop or control worrying
    'q_somatic',             # 0-4: Physical stress symptoms (racing heart, muscle tension)
    'q_sleep_issues',        # 0-4: Trouble falling/staying asleep, unrefreshing sleep
    'q_fatigue',             # 0-4: Feeling tired, drained, lacking energy
    'q_burnout',             # 0-4: Emotionally exhausted, detached from work/study
    'q_concentration',       # 0-4: Brain fog, difficulty concentrating
    'q_isolation',           # 0-4: Feeling lonely or lacking support
    'q_coping_habits',       # 0-4: Regular mindfulness, exercise, restorative habits (reverse distress)
    'sleep_hours',           # Continuous 3.5 - 11.0 hours
    'work_screen_hours'      # Continuous 1.0 - 15.0 hours
]

def generate_mental_health_dataset(n_samples=8000, random_state=42):
    np.random.seed(random_state)
    
    # Latent distress factor (underlying general mental strain distribution)
    latent_distress = np.random.beta(2, 3, size=n_samples) # 0 to 1, slight skew toward moderate/mild
    
    # Generate question responses (0 to 4 Likert scale)
    # Higher distress increases probability of high scores for distress items
    def sample_distress_item(weight, noise_scale=0.15):
        prob_factor = np.clip(latent_distress * weight + np.random.normal(0, noise_scale, size=n_samples), 0, 1)
        # Convert to 0-4
        return np.clip(np.round(prob_factor * 4.0).astype(int), 0, 4)

    def sample_protective_item(weight, noise_scale=0.15):
        # Protective items are inversely correlated with latent distress
        prob_factor = np.clip((1.0 - latent_distress) * weight + np.random.normal(0, noise_scale, size=n_samples), 0, 1)
        return np.clip(np.round(prob_factor * 4.0).astype(int), 0, 4)

    q_anhedonia = sample_distress_item(0.85, 0.18)
    q_depressed = sample_distress_item(0.90, 0.15)
    q_optimism = sample_protective_item(0.88, 0.18) # 4 is very optimistic, 0 is none
    
    q_overwhelmed = sample_distress_item(0.92, 0.14)
    q_uncontrollable = sample_distress_item(0.88, 0.16)
    q_coping_confidence = sample_protective_item(0.85, 0.17)
    
    q_anxious = sample_distress_item(0.90, 0.15)
    q_worry = sample_distress_item(0.87, 0.16)
    q_somatic = sample_distress_item(0.80, 0.20)
    
    q_sleep_issues = sample_distress_item(0.82, 0.20)
    q_fatigue = sample_distress_item(0.86, 0.16)
    q_burnout = sample_distress_item(0.88, 0.17)
    
    q_concentration = sample_distress_item(0.78, 0.20)
    q_isolation = sample_distress_item(0.75, 0.22)
    q_coping_habits = sample_protective_item(0.80, 0.22)
    
    # Context variables correlated with distress
    # Poor sleep duration (<6 or >9.5) correlates with distress
    base_sleep = 7.5 - (latent_distress * 2.2) + np.random.normal(0, 1.0, size=n_samples)
    sleep_hours = np.clip(np.round(base_sleep, 1), 3.5, 11.0)
    
    # High screen/work hours slightly correlates with distress
    base_screen = 5.0 + (latent_distress * 4.5) + np.random.normal(0, 2.0, size=n_samples)
    work_screen_hours = np.clip(np.round(base_screen, 1), 1.0, 15.0)

    # -------------------------------------------------------------
    # Ground Truth Stress Score Calculation (0 - 100, Higher = More Stressed)
    # Grounded in PSS-10 + GAD-7 + Somatic Stress + Burnout
    # -------------------------------------------------------------
    # PSS Core items (uncontrollable, overwhelmed, lacking confidence): max 12
    pss_raw = q_overwhelmed + q_uncontrollable + (4 - q_coping_confidence) # 0 to 12
    pss_contrib = (pss_raw / 12.0) * 35.0 # Up to 35 pts
    
    # Anxiety & Somatic arousal: max 12
    anxiety_raw = q_anxious + q_worry + q_somatic # 0 to 12
    anxiety_contrib = (anxiety_raw / 12.0) * 25.0 # Up to 25 pts
    
    # Burnout & Cognitive load: max 8
    burnout_raw = q_burnout + q_concentration # 0 to 8
    burnout_contrib = (burnout_raw / 8.0) * 20.0 # Up to 20 pts
    
    # Somatic & Sleep strain: max 8
    sleep_strain_raw = q_sleep_issues + q_fatigue # 0 to 8
    sleep_contrib = (sleep_strain_raw / 8.0) * 12.0 # Up to 12 pts
    
    # Lack of restorative coping buffer: max 4
    coping_deficit = (4 - q_coping_habits) # 0 to 4
    coping_contrib = (coping_deficit / 4.0) * 8.0 # Up to 8 pts
    
    # Sleep duration penalty
    sleep_dur_penalty = np.where(sleep_hours < 6.0, (6.0 - sleep_hours) * 2.0,
                        np.where(sleep_hours > 9.5, (sleep_hours - 9.5) * 1.5, 0.0))
    
    # Screen time penalty
    screen_penalty = np.maximum(0, (work_screen_hours - 8.0) * 0.7)
    
    raw_stress_score = (
        pss_contrib + 
        anxiety_contrib + 
        burnout_contrib + 
        sleep_contrib + 
        coping_contrib + 
        sleep_dur_penalty + 
        screen_penalty
    )
    # Add slight random human variance
    stress_score = raw_stress_score + np.random.normal(0, 1.8, size=n_samples)
    stress_score = np.clip(np.round(stress_score, 1), 3.0, 99.0)

    # -------------------------------------------------------------
    # Ground Truth Mental Health Score Calculation (0 - 100, Higher = Better Well-being)
    # Grounded in WHO-5 Wellbeing Index + Positive Psychology Flourishing Scales + Resiliency
    # -------------------------------------------------------------
    # Positive mood & Vitality (Optimism, low anhedonia, low depression):
    mood_positivity = (q_optimism + (4 - q_anhedonia) + (4 - q_depressed)) / 12.0 # 0 to 1
    mood_contrib = mood_positivity * 30.0 # Up to 30 pts
    
    # Emotional Stability & Stress Resilience:
    stress_resilience = 1.0 - (pss_raw / 12.0) # 0 to 1
    resilience_contrib = stress_resilience * 25.0 # Up to 25 pts
    
    # Nervous System Calm & Calmness:
    nervous_calm = 1.0 - (anxiety_raw / 12.0) # 0 to 1
    nervous_contrib = nervous_calm * 18.0 # Up to 18 pts
    
    # Restorative Energy, Sleep & Vitality:
    energy_vitality = 1.0 - (sleep_strain_raw / 8.0)
    sleep_opt_bonus = np.where((sleep_hours >= 7.0) & (sleep_hours <= 8.5), 1.0, 0.0)
    energy_contrib = np.clip(energy_vitality * 12.0 + sleep_opt_bonus * 2.0, 0, 14.0)
    
    # Social Connection & Restorative Habits:
    social_resilience = ((4 - q_isolation) / 4.0) * 6.5
    habits_resilience = (q_coping_habits / 4.0) * 6.5
    social_contrib = social_resilience + habits_resilience # Up to 13 pts
    
    raw_mh_score = mood_contrib + resilience_contrib + nervous_contrib + energy_contrib + social_contrib
    # Subtract slight screen burnout factor if excessive
    raw_mh_score -= np.maximum(0, (work_screen_hours - 9.0) * 0.5)
    
    # Add slight random human variance
    mental_health_score = raw_mh_score + np.random.normal(0, 1.8, size=n_samples)
    mental_health_score = np.clip(np.round(mental_health_score, 1), 5.0, 99.0)

    df = pd.DataFrame({
        'q_anhedonia': q_anhedonia,
        'q_depressed': q_depressed,
        'q_optimism': q_optimism,
        'q_overwhelmed': q_overwhelmed,
        'q_uncontrollable': q_uncontrollable,
        'q_coping_confidence': q_coping_confidence,
        'q_anxious': q_anxious,
        'q_worry': q_worry,
        'q_somatic': q_somatic,
        'q_sleep_issues': q_sleep_issues,
        'q_fatigue': q_fatigue,
        'q_burnout': q_burnout,
        'q_concentration': q_concentration,
        'q_isolation': q_isolation,
        'q_coping_habits': q_coping_habits,
        'sleep_hours': sleep_hours,
        'work_screen_hours': work_screen_hours,
        'mental_health_score': mental_health_score,
        'stress_score': stress_score
    })
    
    return df

def train_and_export_mental_health_models():
    print("Generating representative psychometric dataset...")
    df = generate_mental_health_dataset(n_samples=8000, random_state=42)
    
    X = df[FEATURE_COLS]
    y_mh = df['mental_health_score']
    y_stress = df['stress_score']
    
    X_train, X_test, y_mh_train, y_mh_test, y_st_train, y_st_test = train_test_split(
        X, y_mh, y_stress, test_size=0.15, random_state=42
    )
    
    print("Training Random Forest Regressor for Mental Health Score...")
    mh_model = RandomForestRegressor(
        n_estimators=160,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    mh_model.fit(X_train, y_mh_train)
    
    mh_preds = mh_model.predict(X_test)
    mh_rmse = np.sqrt(mean_squared_error(y_mh_test, mh_preds))
    mh_r2 = r2_score(y_mh_test, mh_preds)
    print(f"Mental Health Model - RMSE: {mh_rmse:.2f}, R2: {mh_r2:.4f}")
    
    print("Training Random Forest Regressor for Stress Score...")
    stress_model = RandomForestRegressor(
        n_estimators=160,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    stress_model.fit(X_train, y_st_train)
    
    st_preds = stress_model.predict(X_test)
    st_rmse = np.sqrt(mean_squared_error(y_st_test, st_preds))
    st_r2 = r2_score(y_st_test, st_preds)
    print(f"Stress Model - RMSE: {st_rmse:.2f}, R2: {st_r2:.4f}")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'mental_health_model.joblib')
    
    # Feature importances
    feature_importance_mh = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in mh_model.feature_importances_]))
    feature_importance_st = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in stress_model.feature_importances_]))
    
    payload = {
        'mh_model': mh_model,
        'stress_model': stress_model,
        'feature_cols': FEATURE_COLS,
        'metrics': {
            'mh_rmse': round(float(mh_rmse), 3),
            'mh_r2': round(float(mh_r2), 4),
            'st_rmse': round(float(st_rmse), 3),
            'st_r2': round(float(st_r2), 4)
        },
        'feature_importances': {
            'mental_health': feature_importance_mh,
            'stress': feature_importance_st
        }
    }
    
    joblib.dump(payload, model_path, compress=3)
    print(f"Mental Health and Stress ML model package saved to: {model_path}")
    return payload

if __name__ == '__main__':
    train_and_export_mental_health_models()
