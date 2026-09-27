"""
ml_engine.py
Health Evaluation & Prediction Engine.
Combines machine learning (RandomForestRegressor) with evidence-based clinical guidelines
(AHA Life's Essential 8, WHO Physical Activity Guidelines, Framingham risk factors, and JNC8 BP categories).
Generates comprehensive category breakdowns, biological age estimation, risk factors,
and tailored 3-phase future action plans.
"""

import os
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'health_model.joblib')

class HealthScorePredictor:
    def __init__(self):
        self.model_data = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found at {MODEL_PATH}. Auto-generating model...")
            try:
                from train_model import train_and_export_model
                train_and_export_model()
            except Exception as e:
                print(f"Notice: Auto-training skipped ({e}). Scoring engine will use clinical algorithmic baseline.")

        if os.path.exists(MODEL_PATH):
            try:
                self.model_data = joblib.load(MODEL_PATH)
                print(f"Loaded ML model from {MODEL_PATH}")
            except Exception as e:
                print(f"Warning: Failed to load model ({e}), using rule-based scoring engine.")
                self.model_data = None
        else:
            self.model_data = None

    def compute_bmi(self, weight_kg, height_cm):
        if height_cm <= 0:
            return 0.0, "Unknown"
        height_m = height_cm / 100.0
        bmi = round(weight_kg / (height_m ** 2), 1)
        
        if bmi < 18.5:
            category = "Underweight"
        elif bmi <= 24.9:
            category = "Normal (Healthy Weight)"
        elif bmi <= 29.9:
            category = "Overweight"
        elif bmi <= 34.9:
            category = "Obesity Class I"
        else:
            category = "Obesity Class II/III"
            
        return bmi, category

    def compute_bp_category(self, systolic, diastolic):
        if systolic < 120 and diastolic < 80:
            return "Normal (<120/80 mmHg)", "optimal"
        elif systolic <= 129 and diastolic < 80:
            return "Elevated (120-129/<80 mmHg)", "warning"
        elif systolic <= 139 or diastolic <= 89:
            return "Stage 1 Hypertension (130-139 / 80-89 mmHg)", "alert"
        elif systolic >= 180 or diastolic >= 120:
            return "Hypertensive Crisis (Urgent attention needed)", "danger"
        else:
            return "Stage 2 Hypertension (>=140 / >=90 mmHg)", "danger"

    def compute_glucose_category(self, glucose):
        if glucose < 70:
            return "Low (Hypoglycemia risk)", "warning"
        elif glucose <= 99:
            return "Normal (<100 mg/dL)", "optimal"
        elif glucose <= 125:
            return "Impaired Fasting Glucose (Prediabetes range: 100-125 mg/dL)", "warning"
        else:
            return "Elevated (Diabetic range: >=126 mg/dL)", "danger"

    def calculate_pillar_scores(self, data, bmi):
        """
        Calculates granular 0-100 scores across 6 core pillars of health:
        1. Cardiovascular Health
        2. Metabolic & Body Composition
        3. Physical Fitness & Activity
        4. Sleep & Circadian Recovery
        5. Mental Wellbeing & Stress
        6. Habits & Lifestyle Cleanliness
        """
        # 1. Cardiovascular Health (BP, resting HR, cholesterol, smoking)
        sys_bp = data.get('systolic_bp', 120)
        dia_bp = data.get('diastolic_bp', 80)
        rhr = data.get('resting_hr', 72)
        chol = data.get('cholesterol', 180)
        smk = data.get('smoking_code', 0)

        cardio_score = 100.0
        # Blood pressure penalty
        if sys_bp > 120:
            cardio_score -= (sys_bp - 120) * 0.8
        if dia_bp > 80:
            cardio_score -= (dia_bp - 80) * 0.8
        # Resting heart rate penalty (optimal: 55-72)
        if rhr > 72:
            cardio_score -= (rhr - 72) * 0.7
        elif rhr < 50:
            cardio_score -= 5 # potential bradycardia unless athletic
        # Cholesterol penalty (desirable < 200)
        if chol > 200:
            cardio_score -= (chol - 200) * 0.25
        # Smoking impact
        cardio_score -= smk * 8.0
        cardio_score = float(np.clip(cardio_score, 10, 100))

        # 2. Metabolic & Body Composition (BMI, Glucose, Diet)
        glucose = data.get('glucose', 90)
        diet = data.get('diet_quality', 3)
        
        metabolic_score = 100.0
        # BMI penalty
        if bmi > 24.9:
            metabolic_score -= (bmi - 24.9) * 2.8
        elif bmi < 18.5:
            metabolic_score -= (18.5 - bmi) * 2.5
        # Glucose penalty
        if glucose > 99:
            metabolic_score -= (glucose - 99) * 0.9
        # Diet contribution
        metabolic_score += (diet - 3) * 5.0
        metabolic_score = float(np.clip(metabolic_score, 10, 100))

        # 3. Physical Fitness & Activity (Activity minutes, Daily steps)
        act_mins = data.get('activity_mins', 150)
        steps = data.get('daily_steps', 7500)
        
        # WHO target: 150-300 min/wk moderate or 75-150 min/wk vigorous; 8000-10000 steps
        act_pts = min(60.0, (act_mins / 200.0) * 60.0)
        step_pts = min(40.0, (steps / 10000.0) * 40.0)
        fitness_score = float(np.clip(act_pts + step_pts, 10, 100))

        # 4. Sleep & Circadian Recovery (Hours, Quality)
        sleep_hrs = data.get('sleep_hours', 7.5)
        sleep_qual = data.get('sleep_quality', 3)
        
        # Optimal sleep: 7.0 to 8.5 hours
        if 7.0 <= sleep_hrs <= 8.5:
            dur_pts = 50.0
        elif 6.0 <= sleep_hrs < 7.0 or 8.5 < sleep_hrs <= 9.5:
            dur_pts = 35.0
        elif 5.0 <= sleep_hrs < 6.0:
            dur_pts = 20.0
        else:
            dur_pts = 10.0
        qual_pts = (sleep_qual / 5.0) * 50.0
        sleep_score = float(np.clip(dur_pts + qual_pts, 10, 100))

        # 5. Mental Wellbeing & Stress (Stress level 1-10, Screen time)
        stress = data.get('stress_level', 5)
        screen = data.get('screen_time', 6.0)
        
        stress_pts = (10 - stress) * 7.5 # 0-75 pts
        screen_pts = max(5.0, 25.0 - max(0, screen - 5) * 3.0) # 0-25 pts
        mental_score = float(np.clip(stress_pts + screen_pts, 10, 100))

        # 6. Habits & Lifestyle Cleanliness (Smoking, Alcohol, Water)
        alc = data.get('alcohol_code', 0)
        water = data.get('water_liters', 2.0)
        
        lifestyle_score = 100.0
        lifestyle_score -= smk * 12.0 # up to -36
        lifestyle_score -= alc * 8.0  # up to -24
        if water < 2.0:
            lifestyle_score -= (2.0 - water) * 10.0
        lifestyle_score = float(np.clip(lifestyle_score, 10, 100))

        return {
            'cardiovascular': round(cardio_score, 1),
            'metabolic': round(metabolic_score, 1),
            'fitness': round(fitness_score, 1),
            'sleep_recovery': round(sleep_score, 1),
            'mental_wellbeing': round(mental_score, 1),
            'habits_lifestyle': round(lifestyle_score, 1)
        }

    def estimate_biological_age(self, chronological_age, pillar_scores, data):
        """
        Estimates biological age based on deviations across cardiovascular,
        metabolic, physical, and lifestyle biomarkers.
        """
        # Average pillar index
        avg_score = np.mean(list(pillar_scores.values()))
        
        # Base delta: baseline score of 75 represents biological age = chronological age
        # Higher score -> younger biological age (up to -8 years)
        # Lower score -> accelerated biological aging (up to +12 years)
        score_delta = (75.0 - avg_score) * 0.22
        
        # Extra specific longevity drivers
        extra_shift = 0.0
        if data.get('smoking_code', 0) >= 2:
            extra_shift += 3.5
        if data.get('systolic_bp', 120) >= 140:
            extra_shift += 2.5
        if data.get('glucose', 90) >= 126:
            extra_shift += 3.0
        if data.get('activity_mins', 150) >= 250 and data.get('daily_steps', 7500) >= 9000:
            extra_shift -= 2.5
        if data.get('sleep_hours', 7.5) >= 7.0 and data.get('stress_level', 5) <= 3:
            extra_shift -= 1.5

        total_shift = score_delta + extra_shift
        # Dampen shift for very young users
        if chronological_age < 25:
            total_shift *= 0.5
            
        biological_age = max(18.0, round(chronological_age + total_shift, 1))
        diff = round(biological_age - chronological_age, 1)
        
        if diff <= -2.0:
            status = "Youthful / Optimal Vitality"
            comment = f"Your biomarkers suggest your physiological vitality is roughly {abs(diff):.1f} years younger than your calendar age!"
        elif diff < 2.0:
            status = "Age-Matched Healthy"
            comment = "Your physiological age aligns closely with your chronological age."
        elif diff < 5.0:
            status = "Mild Accelerated Aging"
            comment = f"Your biological age shows an estimated +{diff:.1f} years advance, primarily due to manageable lifestyle or metabolic factors."
        else:
            status = "Elevated Biological Aging"
            comment = f"Your biological age reflects an estimated +{diff:.1f} years advance. Implementing targeted interventions will quickly restore vitality."

        return {
            'chronological_age': chronological_age,
            'biological_age': biological_age,
            'difference': diff,
            'status': status,
            'comment': comment
        }

    def evaluate_risk_profile(self, data, bmi):
        risks = []
        
        # Cardiovascular risk
        cvd_risk_pts = 0
        if data.get('systolic_bp', 120) >= 140 or data.get('diastolic_bp', 80) >= 90:
            cvd_risk_pts += 3
        elif data.get('systolic_bp', 120) >= 130 or data.get('diastolic_bp', 80) >= 85:
            cvd_risk_pts += 1.5
        if data.get('cholesterol', 180) >= 230:
            cvd_risk_pts += 2
        if data.get('smoking_code', 0) >= 2:
            cvd_risk_pts += 3
        if data.get('family_cvd', 0) == 1:
            cvd_risk_pts += 1.5
        if data.get('resting_hr', 70) >= 85:
            cvd_risk_pts += 1
            
        if cvd_risk_pts >= 5:
            cvd_tier = "High"
            cvd_desc = "Multiple elevated cardiovascular risk markers detected (BP, lipids, or habits)."
        elif cvd_risk_pts >= 2.5:
            cvd_tier = "Moderate"
            cvd_desc = "Mild cardiovascular risk factors identified. Highly reversible with proactive lifestyle changes."
        else:
            cvd_tier = "Low"
            cvd_desc = "Cardiovascular biomarkers fall within healthy protective thresholds."

        risks.append({
            'name': 'Cardiovascular Risk',
            'level': cvd_tier,
            'badge': 'danger' if cvd_tier == 'High' else ('warning' if cvd_tier == 'Moderate' else 'optimal'),
            'description': cvd_desc
        })

        # Metabolic & Type-2 Diabetes Risk
        met_pts = 0
        if bmi >= 30:
            met_pts += 3
        elif bmi >= 25:
            met_pts += 1.5
        if data.get('glucose', 90) >= 126:
            met_pts += 4
        elif data.get('glucose', 90) >= 100:
            met_pts += 2
        if data.get('family_diabetes', 0) == 1:
            met_pts += 1.5
        if data.get('activity_mins', 150) < 60:
            met_pts += 1.5

        if met_pts >= 5:
            met_tier = "High"
            met_desc = "High metabolic strain / insulin resistance risk detected. Focus on blood glucose stabilization."
        elif met_pts >= 2.5:
            met_tier = "Moderate"
            met_desc = "Subtle signs of metabolic friction. Targeted nutrition and movement will provide rapid optimization."
        else:
            met_tier = "Low"
            met_desc = "Metabolic markers and glycemic baseline are stable and well-regulated."

        risks.append({
            'name': 'Metabolic / Glycemic Risk',
            'level': met_tier,
            'badge': 'danger' if met_tier == 'High' else ('warning' if met_tier == 'Moderate' else 'optimal'),
            'description': met_desc
        })

        # Burnout & Sleep Debt Risk
        stress_pts = 0
        if data.get('stress_level', 5) >= 8:
            stress_pts += 3
        elif data.get('stress_level', 5) >= 6:
            stress_pts += 1.5
        if data.get('sleep_hours', 7.5) < 6.0:
            stress_pts += 2.5
        if data.get('sleep_quality', 3) <= 2:
            stress_pts += 1.5
        if data.get('screen_time', 6.0) >= 10:
            stress_pts += 1.5

        if stress_pts >= 5:
            burnout_tier = "Elevated"
            burnout_desc = "High allostatic load and sleep deficit. Nervous system requires intentional downregulation."
        elif stress_pts >= 2.5:
            burnout_tier = "Moderate"
            burnout_desc = "Moderate stress and recovery imbalance. Evening screen boundaries and circadian focus recommended."
        else:
            burnout_tier = "Balanced"
            burnout_desc = "Healthy restorative balance and adaptive stress recovery."

        risks.append({
            'name': 'Stress & Burnout Risk',
            'level': burnout_tier,
            'badge': 'danger' if burnout_tier == 'Elevated' else ('warning' if burnout_tier == 'Moderate' else 'optimal'),
            'description': burnout_desc
        })

        return risks

    def generate_future_action_plan(self, data, bmi, pillars, overall_score):
        """
        Creates an actionable, 3-phase evidence-based personal health roadmap:
        - Phase 1: Days 1 - 14 (Immediate Micro-Habit Quick Wins)
        - Phase 2: Days 15 - 45 (Habit Solidification & Routine Architecture)
        - Phase 3: Days 46 - 90 (Measurable Transformation & Milestones)
        - Clinical Diagnostic & Screening Checklist
        """
        quick_wins = []
        routine_plan = []
        long_term_goals = []
        clinical_actions = []

        # Find weakest pillars to customize advice
        sorted_pillars = sorted(pillars.items(), key=lambda x: x[1])
        weakest_area = sorted_pillars[0][0]
        second_weakest = sorted_pillars[1][0]

        # 1. Quick Wins (Days 1 - 14)
        if pillars['sleep_recovery'] < 70 or data.get('sleep_hours', 7.5) < 7.0:
            quick_wins.append({
                'title': 'Nightly Digital Wind-Down Curfew',
                'category': 'Sleep & Recovery',
                'icon': 'fa-moon',
                'action': 'Turn off or switch all screens to warm night-mode 45 minutes before sleep. Implement a consistent bedtime within a 30-minute window.'
            })
            quick_wins.append({
                'title': 'Morning Sunlight Exposure',
                'category': 'Circadian Rhythm',
                'icon': 'fa-sun',
                'action': 'Get 10–15 minutes of direct morning outdoor light within 1 hour of waking to anchor cortisol and elevate nighttime melatonin secretion.'
            })

        if pillars['cardiovascular'] < 70 or data.get('systolic_bp', 120) >= 130:
            quick_wins.append({
                'title': 'Sodium Reduction & Hydration Balancing',
                'category': 'Cardiovascular',
                'icon': 'fa-heart-pulse',
                'action': 'Limit processed convenience foods and high-sodium sauces; aim for at least 2.5L water daily to relieve arterial tension.'
            })
            quick_wins.append({
                'title': 'Daily Post-Meal Brisk 10-Minute Walk',
                'category': 'Cardio & Glucose',
                'icon': 'fa-person-walking',
                'action': 'Take an easy 10-minute stroll after lunch and dinner. Studies show this lowers postprandial glucose surges by up to 25%.'
            })

        if pillars['fitness'] < 65 or data.get('daily_steps', 7500) < 6000:
            quick_wins.append({
                'title': 'Step Increment Baseline (+2,000 steps)',
                'category': 'Physical Activity',
                'icon': 'fa-shoe-prints',
                'action': 'Add a simple 15-minute dedicated morning or evening walk to your baseline routine to build effortless momentum.'
            })

        if data.get('water_liters', 2.0) < 2.0:
            quick_wins.append({
                'title': 'Strategic Morning Hydration Protocol',
                'category': 'Hydration',
                'icon': 'fa-glass-water',
                'action': 'Drink a 500ml glass of water immediately upon waking to kickstart metabolic clearance and rehydrate vital tissues.'
            })

        if pillars['mental_wellbeing'] < 65 or data.get('stress_level', 5) >= 7:
            quick_wins.append({
                'title': 'Box Breathing & Decompression Reset',
                'category': 'Stress Resilience',
                'icon': 'fa-lungs',
                'action': 'Practice 4-4-4-4 Box Breathing (inhale 4s, hold 4s, exhale 4s, hold 4s) for 5 minutes midday or during stress spikes.'
            })

        # Ensure at least 3-4 high value quick wins
        if len(quick_wins) < 3:
            quick_wins.append({
                'title': 'Color Diversity Nutrition Habit',
                'category': 'Nutrition',
                'icon': 'fa-apple-whole',
                'action': 'Incorporate at least 2 distinct colorful vegetables or berries into your main meals to boost polyphenol intake.'
            })

        # 2. Phase 2: Routine Architecture (Days 15 - 45)
        routine_plan.append({
            'pillar': 'Structured Physical Fitness',
            'frequency': '3 - 4 sessions per week',
            'detail': 'Combine 150 minutes of Zone 2 cardio (conversational pace jogging, cycling, brisk uphill walking) with 2 full-body resistance training sessions focusing on compound movements.'
        })

        if bmi >= 25 or data.get('glucose', 90) >= 100:
            routine_plan.append({
                'pillar': 'Nutritional Protocol: Mediterranean / DASH Pattern',
                'frequency': 'Daily meals',
                'detail': 'Prioritize lean proteins, complex fiber (legumes, oats, leafy greens), healthy fats (extra virgin olive oil, avocado, walnuts), and minimize refined sugars and ultra-processed carbohydrates.'
            })
        else:
            routine_plan.append({
                'pillar': 'Metabolic Nutrition & Sustained Energy',
                'frequency': 'Daily meals',
                'detail': 'Ensure 25-35g of dietary fiber and adequate high-quality protein per meal (1.2-1.6g/kg of ideal bodyweight) to preserve muscle mass and regulate hunger signals.'
            })

        routine_plan.append({
            'pillar': 'Circadian Sleep Hygiene Framework',
            'frequency': 'Every evening',
            'detail': 'Keep bedroom cool (18-20°C / 65-68°F), eliminate ambient LED light sources, avoid caffeine within 9 hours of bedtime, and avoid heavy meals within 3 hours of sleep.'
        })

        if data.get('smoking_code', 0) > 0 or data.get('alcohol_code', 0) >= 2:
            routine_plan.append({
                'pillar': 'Habit Optimization & Toxic Burden Reduction',
                'frequency': 'Weekly tracking',
                'detail': 'Institute alcohol-free weekdays and a progressive tapering schedule for nicotine or recreational substances with professional smoking cessation resources.'
            })

        # 3. Phase 3: Long-term Transformation Goals (Days 46 - 90)
        target_bp = "118 / 78 mmHg" if data.get('systolic_bp', 120) > 125 else "Maintain <120 / <80 mmHg"
        target_rhr = "60 - 68 bpm" if data.get('resting_hr', 70) > 72 else "Maintain current athletic baseline"
        
        long_term_goals.append({
            'milestone': 'Cardiovascular Vitality Benchmark',
            'metric': f'Target Blood Pressure: {target_bp} | Target Resting HR: {target_rhr}',
            'target_timeframe': 'Day 60 - 90 Checkpoint',
            'impact': 'Lowers 10-year major adverse cardiac event probability by up to 35%.'
        })

        if bmi > 24.9:
            target_weight_loss = round(data.get('weight', 75) * 0.05, 1) # 5% loss target
            long_term_goals.append({
                'milestone': 'Metabolic Reset & Body Composition',
                'metric': f'Gradual, sustainable 5% body mass reduction (~{target_weight_loss} kg)',
                'target_timeframe': '90-Day Target',
                'impact': 'Restores insulin sensitivity, reduces visceral adipose tissue, and reduces liver enzymes.'
            })

        long_term_goals.append({
            'milestone': 'Functional Aerobic Base & Muscular Resilience',
            'metric': 'Achieve consistently >8,500 daily steps and 20+ unbroken bodyweight pushups / 60s plank',
            'target_timeframe': 'Day 90 Evaluation',
            'impact': 'Strongest clinical predictor of all-cause longevity and physical independence.'
        })

        # 4. Clinical & Screening Checklist
        if data.get('systolic_bp', 120) >= 135 or data.get('diastolic_bp', 80) >= 88:
            clinical_actions.append({
                'test': 'Home Blood Pressure Monitoring Log',
                'urgency': 'Within 2 weeks',
                'reason': 'Record twice-daily BP (morning and evening) for 7 consecutive days and share with your primary care provider for formal clinical staging.'
            })

        if data.get('glucose', 90) >= 100 or data.get('family_diabetes', 0) == 1 or bmi >= 28:
            clinical_actions.append({
                'test': 'Fasting Blood Glucose + HbA1c Panel',
                'urgency': 'Next routine checkup',
                'reason': 'Measures your average 3-month glycemic control to rule out prediabetes or insulin resistance.'
            })

        if data.get('cholesterol', 180) >= 200 or data.get('family_cvd', 0) == 1 or data.get('age', 35) >= 40:
            clinical_actions.append({
                'test': 'Comprehensive Lipid Panel (Total, LDL-C, HDL-C, Triglycerides, ApoB)',
                'urgency': 'Recommended within 1-3 months',
                'reason': 'Accurately quantifies atherogenic particle burden and guides proactive cardiovascular protection.'
            })

        clinical_actions.append({
            'test': 'Comprehensive Metabolic Panel (CMP) + Serum Vitamin D (25-OH)',
            'urgency': 'Annual preventative screening',
            'reason': 'Assesses renal, hepatic, electrolyte, and immune/bone metabolic homeostasis.'
        })

        return {
            'quick_wins': quick_wins,
            'routine_plan': routine_plan,
            'long_term_goals': long_term_goals,
            'clinical_actions': clinical_actions
        }

    def predict(self, raw_input):
        """
        Main evaluation pipeline.
        Accepts dictionary of user health attributes, validates/computes derived metrics,
        runs ML model prediction + clinical rule ensemble, and generates complete action plan.
        """
        # Parse & sanitize inputs
        age = int(raw_input.get('age', 35))
        gender = raw_input.get('gender', 'male')
        gender_code = 1 if gender.lower() == 'male' else 0
        height = float(raw_input.get('height', 170.0))
        weight = float(raw_input.get('weight', 70.0))

        bmi, bmi_category = self.compute_bmi(weight, height)

        systolic_bp = float(raw_input.get('systolic_bp', 120.0))
        diastolic_bp = float(raw_input.get('diastolic_bp', 80.0))
        resting_hr = float(raw_input.get('resting_hr', 72.0))
        glucose = float(raw_input.get('glucose', 90.0))
        cholesterol = float(raw_input.get('cholesterol', 185.0))

        activity_mins = float(raw_input.get('activity_mins', 150.0))
        daily_steps = float(raw_input.get('daily_steps', 7500.0))
        sleep_hours = float(raw_input.get('sleep_hours', 7.5))
        sleep_quality = int(raw_input.get('sleep_quality', 3))
        water_liters = float(raw_input.get('water_liters', 2.2))
        diet_quality = int(raw_input.get('diet_quality', 3))

        smoking_str = str(raw_input.get('smoking_status', 'never')).lower()
        smoking_map = {'never': 0, 'former': 1, 'occasional': 2, 'regular': 3}
        smoking_code = smoking_map.get(smoking_str, 0)

        alcohol_str = str(raw_input.get('alcohol_intake', 'light')).lower()
        alcohol_map = {'none': 0, 'light': 1, 'moderate': 2, 'heavy': 3}
        alcohol_code = alcohol_map.get(alcohol_str, 1)

        stress_level = int(raw_input.get('stress_level', 5))
        screen_time = float(raw_input.get('screen_time', 6.0))

        family_cvd = 1 if raw_input.get('family_cvd') in [True, 1, '1', 'true', 'True'] else 0
        family_diabetes = 1 if raw_input.get('family_diabetes') in [True, 1, '1', 'true', 'True'] else 0
        family_hypertension = 1 if raw_input.get('family_hypertension') in [True, 1, '1', 'true', 'True'] else 0

        feature_dict = {
            'age': age,
            'gender_code': gender_code,
            'height': height,
            'weight': weight,
            'bmi': bmi,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'resting_hr': resting_hr,
            'glucose': glucose,
            'cholesterol': cholesterol,
            'activity_mins': activity_mins,
            'daily_steps': daily_steps,
            'sleep_hours': sleep_hours,
            'sleep_quality': sleep_quality,
            'water_liters': water_liters,
            'diet_quality': diet_quality,
            'smoking_code': smoking_code,
            'alcohol_code': alcohol_code,
            'stress_level': stress_level,
            'screen_time': screen_time,
            'family_cvd': family_cvd,
            'family_diabetes': family_diabetes,
            'family_hypertension': family_hypertension
        }

        # Compute granular pillars
        pillars = self.calculate_pillar_scores(feature_dict, bmi)
        
        # ML Score Prediction
        if self.model_data is not None:
            try:
                import pandas as pd
                feature_order = self.model_data['feature_cols']
                df_features = pd.DataFrame([{col: feature_dict[col] for col in feature_order}])
                ml_score = float(self.model_data['model'].predict(df_features)[0])
            except Exception as e:
                print(f"Prediction fallback: {e}")
                ml_score = np.mean(list(pillars.values()))
        else:
            ml_score = np.mean(list(pillars.values()))

        # Blend ML Score with weighted pillar evaluation
        weights = {
            'cardiovascular': 0.25,
            'metabolic': 0.22,
            'fitness': 0.18,
            'sleep_recovery': 0.15,
            'mental_wellbeing': 0.10,
            'habits_lifestyle': 0.10
        }
        pillar_weighted = sum(pillars[k] * weights[k] for k in weights)
        final_score = round(0.55 * ml_score + 0.45 * pillar_weighted, 1)
        final_score = float(np.clip(final_score, 15.0, 99.0))

        # Overall Status Tier
        if final_score >= 88.0:
            status_tier = "Optimal Vitality"
            status_color = "#10b981" # Emerald
            badge_class = "optimal"
            summary_quote = "Outstanding biomarker and lifestyle harmony. You are in peak longevity condition."
        elif final_score >= 74.0:
            status_tier = "Good / Robust Baseline"
            status_color = "#3b82f6" # Blue
            badge_class = "good"
            summary_quote = "Solid overall health foundation with minor opportunities for targeted refinement."
        elif final_score >= 58.0:
            status_tier = "Moderate / Needs Optimization"
            status_color = "#f59e0b" # Amber
            badge_class = "warning"
            summary_quote = "Multiple lifestyle or biomarker stressors detected. Implementing the action plan will yield fast gains."
        else:
            status_tier = "Elevated Risk / Action Required"
            status_color = "#ef4444" # Red
            badge_class = "danger"
            summary_quote = "Key indicators highlight significant metabolic, cardiovascular, or lifestyle strain. Timely intervention is advised."

        # Derived metrics
        bp_label, bp_status = self.compute_bp_category(systolic_bp, diastolic_bp)
        glucose_label, glucose_status = self.compute_glucose_category(glucose)
        bio_age_data = self.estimate_biological_age(age, pillars, feature_dict)
        risk_profile = self.evaluate_risk_profile(feature_dict, bmi)
        action_plan = self.generate_future_action_plan(feature_dict, bmi, pillars, final_score)

        return {
            'overall_score': final_score,
            'status_tier': status_tier,
            'status_color': status_color,
            'badge_class': badge_class,
            'summary_quote': summary_quote,
            'bmi': bmi,
            'bmi_category': bmi_category,
            'bp_label': bp_label,
            'bp_status': bp_status,
            'glucose_label': glucose_label,
            'glucose_status': glucose_status,
            'biological_age': bio_age_data,
            'pillars': pillars,
            'risk_profile': risk_profile,
            'action_plan': action_plan,
            'input_echo': {
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'resting_hr': resting_hr,
                'glucose': glucose,
                'cholesterol': cholesterol,
                'activity_mins': activity_mins,
                'daily_steps': daily_steps,
                'sleep_hours': sleep_hours,
                'sleep_quality': sleep_quality,
                'water_liters': water_liters,
                'diet_quality': diet_quality,
                'smoking_status': smoking_str,
                'alcohol_intake': alcohol_str,
                'stress_level': stress_level,
                'screen_time': screen_time,
                'family_cvd': family_cvd,
                'family_diabetes': family_diabetes,
                'family_hypertension': family_hypertension
            }
        }
