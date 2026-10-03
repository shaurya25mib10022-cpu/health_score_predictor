# MindVitalis — AI Mental Health, Stress & Physical Health Predictor

A clinical-grade, machine-learning-powered **Mental Health & Stress Score Predictor** and **Longevity Predictor** application. Evaluates user responses to evidence-based psychometric questions (PSS-10, GAD-7, PHQ-9, WHO-5, Maslach Burnout) to predict an overall **Mental Health Score (0–100)** and **Stress Score (0–100)**, renders multi-domain radar charts, flags critical strain triggers, and provides an interactive 4-7-8 breathing calm widget alongside a personalized 3-phase coping roadmap.

---

## 🌟 Key Features

### 🧠 1. AI Mental Health & Stress Score Predictor
- **Evidence-Based Questionnaire**: 15 validated screening questions + circadian & screen-time inputs:
  1. **Mood & Emotional Vitality (PHQ / WHO-5)**: Anhedonia, persistent low mood/depression, cheerfulness & optimism.
  2. **Stress & Perceived Control (PSS-10)**: Feeling overwhelmed by difficulties, feeling unable to control events, coping self-efficacy.
  3. **Anxiety & Nervous Tension (GAD-7)**: Nervousness & on-edge feelings, uncontrollable worry loops, somatic tension (palpitations, muscle stiffness, headaches).
  4. **Sleep & Restorative Recovery**: Insomnia/night awakenings, daytime mental exhaustion, average hours of sleep per night.
  5. **Cognitive Vitality & Burnout (Maslach Scale)**: Emotional detachment/burnout from duties, brain fog/indecision, daily screen/focus hours.
  6. **Social Connection & Coping Buffers**: Social isolation/lack of confidants, active stress recovery habits (exercise, hobbies, mindfulness).

- **Dual Machine Learning Models (Random Forest Regressors, $R^2 \approx 0.96$)**:
  - **Mental Health Score (0 – 100)**: Higher indicates greater psychological flourishing, emotional stability, and resilience.
    - *85–100*: Flourishing & Optimal Well-Being
    - *70–84*: Resilient & Balanced
    - *50–69*: Moderate / Mild Strain
    - *30–49*: Elevated Distress / Vulnerability
    - *0–29*: High Clinical Risk / Severe Strain
  - **Stress Score (0 – 100)**: Higher indicates elevated allostatic load and nervous system activation.
    - *0–25*: Low / Serene Eustress
    - *26–50*: Mild / Moderate Daily Stress
    - *51–75*: High / Elevated Chronic Stress
    - *76–100*: Critical / Severe Acute Overwhelm

- **6 Psychometric Domains & Radar Visualization**:
  - Emotional Vitality & Mood
  - Stress Control & Resilience
  - Nervous Regulation & Calm
  - Sleep & Restorative Recovery
  - Cognitive Vitality & Burnout Resistance
  - Social Buffer & Coping Habits

- **Interactive 4-7-8 Parasympathetic Breathing Calm Widget**:
  - Interactive on-screen visual breathing bubble guiding users through:
    - *Inhale (4s)*: Expands circle, stimulates diaphragmatic breathing.
    - *Hold (7s)*: Gently holds breath, stabilizes heart rhythm.
    - *Exhale (8s)*: Slow mouth exhale triggering the parasympathetic brake via vagus nerve stimulation.
  - Live seconds countdown and cycle tracker with Start/Pause controls.

- **Actionable 3-Phase Coping Roadmap & Crisis Helplines**:
  - *Phase 1 (Days 1–7)*: Autonomic Decompression & Sleep Stabilization
  - *Phase 2 (Weeks 2–4)*: Cognitive Boundaries & Cortisol Regulation
  - *Phase 3 (Month 2+)*: Sustainable Long-term Psychological Fitness & Flourishing
  - *Emergency Helplines*: 988 (US/Canada), 741741 (Crisis Text Line), 1800-599-0019 (KIRAN India), 111 (NHS UK), Befrienders Worldwide.

---

### ❤️ 2. Physical Health & Longevity Predictor (Preserved)
- Seamlessly toggle to the **Physical Health** tab to evaluate vitals, BMI, blood pressure categories, fasting glucose, cholesterol, biological age estimation, and cardiovascular risk stratification.

---

## 🚀 Quick Start Guide

### 1. Requirements & Setup
Ensure Python 3.10+ is installed with the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Terminal CLI
You can take the psychological assessment directly in your command line:
```bash
python mental_health_cli.py
```
This prompts you through each question interactively and prints a formatted terminal report showing your Mental Health Score, Stress Score, domain bars, red flags, strengths, and coping steps!

### 3. Run the Web Application
Start the Flask web server:
```bash
python app.py
```
Then open your browser at:
```
http://localhost:5000
```
- Select preset personas (*🌱 Thriving*, *💼 Work Burnout*, *😰 High Anxiety*, *🌧️ Overwhelmed*) for instant 1-click evaluation, or fill out the questionnaire.
- Switch between **Mental Health & Stress** and **Physical Health** using the top navigation switcher.
- Click **Print / Save PDF** to export your formatted report.

---

## 📡 REST API Documentation

### Mental Health Endpoints

#### 1. `POST /api/mental-health/predict`
Calculates Mental Health Score and Stress Score from user questionnaire inputs.
```json
{
  "q_anhedonia": 1,
  "q_depressed": 1,
  "q_optimism": 3,
  "q_overwhelmed": 1,
  "q_uncontrollable": 1,
  "q_coping_confidence": 3,
  "q_anxious": 1,
  "q_worry": 1,
  "q_somatic": 1,
  "q_sleep_issues": 1,
  "q_fatigue": 1,
  "q_burnout": 1,
  "q_concentration": 1,
  "q_isolation": 1,
  "q_coping_habits": 2,
  "sleep_hours": 7.5,
  "work_screen_hours": 6.5
}
```
**Response:**
```json
{
  "status": "success",
  "data": {
    "mental_health_score": 88.5,
    "stress_score": 15.2,
    "mental_health_tier": {
      "title": "Flourishing & Optimal Well-Being",
      "badge": "Optimal",
      "color": "emerald"
    },
    "stress_tier": {
      "title": "Low / Serene Stress Level",
      "badge": "Serene / Low",
      "color": "emerald"
    },
    "domains": { ... },
    "risk_factors": [ ... ],
    "strengths": [ ... ],
    "action_plan": { ... }
  }
}
```

#### 2. `GET /api/mental-health/presets`
Returns all 4 pre-configured demo personas.

#### 3. `GET /api/mental-health/questions`
Returns metadata and option choices for all 15 screening questions.

---

## 🧪 Running Automated Tests
Run the comprehensive test suite verifying the ML engine, psychometric formulas, API routes, and personas:
```bash
python test_app.py
```

---

## ⚕️ Psychological & Medical Disclaimer
MindVitalis AI is an educational predictive modeling and psychometric self-screening tool. It is not intended to provide clinical psychiatric diagnoses, replace therapy, or prescribe treatment. If you are experiencing acute distress, suicidal ideation, or severe mental health difficulties, please reach out immediately to your local emergency medical service or call a crisis hotline (such as dialing **988** in the US/Canada or **1800-599-0019** in India).
