# Vitalis AI — Health Score Predictor & Future Action Planner

A clinical-grade, machine-learning-powered **Health Score & Longevity Predictor** web application that evaluates holistic multi-biomarker inputs, estimates physiological biological age vs chronological age, analyzes cardiovascular and metabolic risk stratification, and provides a 3-phase evidence-based future action roadmap.

---

## 🌟 Key Features

### 1. Multi-Dimensional Biomarker & Lifestyle Input
- **Demographics & Anthropometrics**: Age, Biological Sex, Height, Weight with live real-time BMI classification.
- **Cardiovascular & Clinical Vitals**: Systolic & Diastolic Blood Pressure (with real-time AHA stage classification), Resting Heart Rate, Fasting Blood Glucose, and Total Serum Cholesterol.
- **Physical Activity & Movement**: Weekly moderate-to-vigorous exercise minutes (WHO benchmarked) and daily step volume.
- **Restorative Sleep & Circadian Health**: Sleep duration, self-reported sleep architecture quality (1–5 scale), and daily screen time.
- **Hydration & Nutritional Pattern**: Liters of water daily and dietary quality (Mediterranean/whole food rating).
- **Stress & Toxic Habits**: Perceived allostatic stress scale (1–10 slider), smoking/nicotine status, and alcohol frequency.
- **First-Degree Family Genetics**: Early cardiovascular disease, type 2 diabetes, and chronic hypertension.

### 2. Machine Learning & Clinical Ensemble Engine
- **Random Forest Regressor**: Trained on a 6,000-sample multivariate clinical dataset grounded in NHANES, Framingham Heart Study, and AHA Life's Essential 8 metrics ($R^2 > 0.82$).
- **Granular 6-Pillar Health Score (0 - 100)**:
  1. *Cardiovascular Health*
  2. *Metabolic & Body Composition*
  3. *Physical Fitness & Activity*
  4. *Sleep & Circadian Recovery*
  5. *Mental Wellbeing & Stress Resilience*
  6. *Habits & Lifestyle Cleanliness*
- **Biological Age Estimation**: Calculates physiological longevity delta based on metabolic, arterial, and lifestyle markers (e.g. *32 calendar age -> 27.5 biological age*).
- **Risk Stratification Matrix**: Low, Moderate, or High tier classification for Cardiovascular Disease, Insulin Resistance / Type 2 Diabetes, and Burnout / Chronic Fatigue.

### 3. Dynamic 3-Phase Future Action Roadmap
- **Phase 1: Days 1 – 14 (Immediate Micro-Habits & Quick Wins)**: Interactive checklist of high-yield habits targeting the user's lowest-scoring categories (e.g., post-meal 10-min walk, morning sunlight anchor, screen wind-down).
- **Phase 2: Days 15 – 45 (Lifestyle Routine Architecture)**: Weekly schedules for Zone 2 cardio, resistance training, Mediterranean nutrition, and sleep hygiene.
- **Phase 3: Days 46 – 90 (Measurable Longevity Milestones)**: Hard clinical target metrics (Target BP <120/80 mmHg, Target Resting HR, 5% metabolic weight reduction).
- **Recommended Diagnostic Screenings**: Specific tests to discuss with a physician (Comprehensive Lipid Panel / ApoB, Fasting Glucose + HbA1c, CMP, Vitamin D).

### 4. Modern, Responsive Frontend
- **Medical-Tech Design**: Tailwind CSS, Dark / Light mode toggle with local storage persistence, responsive mobile/desktop layout.
- **Interactive Visualizations**:
  - Animated SVG circular gauge for the composite vitality score.
  - Interactive **Chart.js** 6-axis Radar Chart comparing individual scores against optimal longevity benchmarks.
  - Quick 1-click **Preset Personas** (*🏃 Endurance Athlete*, *💼 Desk Worker*, *⚠️ High Metabolic Risk*, *🧘 Senior Pro*).
- **Print / PDF Summary**: Formatted medical report export for physical print or PDF saving.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (Tested with Python 3.13)
- Required packages: `flask`, `scikit-learn`, `pandas`, `numpy`, `joblib`

### 2. Installation
Clone or navigate to the directory and install dependencies:
```bash
cd c:\Users\shara\Downloads\health_score_predictor
pip install -r requirements.txt
```

### 3. Train or Verify Model (Pre-trained included)
To train a fresh Random Forest model:
```bash
python train_model.py
```
*(The trained model is exported to `models/health_model.joblib`)*

### 4. Run the Application
Start the Flask web server:
```bash
python app.py
```

Then open your browser at:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 📁 Project Structure

```
health_score_predictor/
├── app.py                     # Production Flask web server & REST API
├── ml_engine.py               # ML scoring engine, pillar analytics, biological age calculator
├── train_model.py             # Synthetic cohort generator & RandomForest training script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── models/
│   └── health_model.joblib    # Serialized trained model & feature metadata
├── templates/
│   └── index.html             # Responsive frontend dashboard
└── static/
    ├── css/
    │   └── styles.css         # Custom animations, range styles, print styles
    └── js/
        └── app.js             # UI logic, Chart.js radar, dynamic BMI/BP calculations
```

---

## 📡 REST API Documentation

### `POST /api/predict`
Calculates comprehensive health score and recommendations.

**Request Payload:**
```json
{
  "age": 32,
  "gender": "male",
  "height": 178,
  "weight": 74,
  "systolic_bp": 118,
  "diastolic_bp": 76,
  "resting_hr": 62,
  "glucose": 88,
  "cholesterol": 175,
  "activity_mins": 220,
  "daily_steps": 9500,
  "sleep_hours": 7.8,
  "sleep_quality": 4,
  "water_liters": 2.8,
  "diet_quality": 4,
  "smoking_status": "never",
  "alcohol_intake": "light",
  "stress_level": 3,
  "screen_time": 5.5,
  "family_cvd": false,
  "family_diabetes": false,
  "family_hypertension": false
}
```

**Response Format:**
```json
{
  "status": "success",
  "data": {
    "overall_score": 85.6,
    "status_tier": "Good / Robust Baseline",
    "bmi": 23.4,
    "bmi_category": "Normal (Healthy Weight)",
    "bp_label": "Normal (<120/80 mmHg)",
    "biological_age": {
      "chronological_age": 32,
      "biological_age": 27.2,
      "difference": -4.8,
      "status": "Youthful / Optimal Vitality"
    },
    "pillars": {
      "cardiovascular": 94.0,
      "metabolic": 91.0,
      "fitness": 96.0,
      "sleep_recovery": 85.0,
      "mental_wellbeing": 80.0,
      "habits_lifestyle": 95.0
    },
    "risk_profile": [...],
    "action_plan": {
      "quick_wins": [...],
      "routine_plan": [...],
      "long_term_goals": [...],
      "clinical_actions": [...]
    }
  }
}
```

---

## ⚕️ Medical Disclaimer
This software is built for educational, predictive wellness demonstration and informational purposes only. It is not intended to diagnose, treat, cure, or prevent any medical condition. Users should always consult a licensed medical professional or physician for health decisions.
