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



## ⚕️ Medical Disclaimer
This software is built for educational, predictive wellness demonstration and informational purposes only. It is not intended to diagnose, treat, cure, or prevent any medical condition. Users should always consult a licensed medical professional or physician for health decisions.
